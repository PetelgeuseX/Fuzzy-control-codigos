"""
HMI - Control Fuzzy de velocidad de motor DC  (SIMULACIÓN + REAL en una sola app)

Modos:
  * Simulación : motor DC modelado como sistema de 1er orden con zona muerta,
                 cuantización del encoder y perturbación de carga. Usa EXACTAMENTE
                 el mismo controlador fuzzy que el firmware del ESP32.
  * Real       : comunicación serial con el ESP32 (fuzzy_mejorado.ino).

Extras:
  * Métricas del último escalón: sobreimpulso, tiempo de establecimiento (2 %),
    error en estado estable, IAE e ITAE.
  * Exportar datos a CSV.
  * python fuzzy_hmi.py --figuras  ->  genera PNG de membresías y superficie de
    control (requiere matplotlib) para el artículo.

Dependencias:  pip install PySide6 pyqtgraph pyserial numpy  (matplotlib opcional)
"""

import sys
import time
import csv
import math
from bisect import bisect_left
from collections import deque

import numpy as np

# ======================================================================
# NÚCLEO (sin interfaz): controlador fuzzy + modelo del motor
# Debe coincidir con fuzzy_mejorado.ino
# ======================================================================

E_MAX = 100.0           # universo del error [RPM]
DE_MAX = 10.0           # universo de dError [RPM/muestra]
KU_DEFECTO = 0.5        # ganancia de salida
RPM_MAX = 100.0
PULSOS_POR_VUELTA = 858.0
TS = 0.02               # periodo de control [s]
N_VENTANA = 5           # muestras de la ventana de medición de RPM

# Filas = E, columnas = dE ; 0=NB 1=NS 2=ZE 3=PS 4=PB
REGLAS = [
    [0, 0, 0, 1, 2],
    [0, 1, 1, 2, 3],
    [0, 1, 2, 3, 4],
    [1, 2, 3, 3, 4],
    [2, 3, 4, 4, 4],
]

ETIQUETAS = ["NB", "NS", "ZE", "PS", "PB"]
COLORES_ET = ["#1e88e5", "#26a69a", "#b0bec5", "#fb8c00", "#e53935"]


def left_shoulder(x, a, b):
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)


def right_shoulder(x, a, b):
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


def trimf(x, a, b, c):
    if x <= a or x >= c:
        return 1.0 if x == b else 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def membresias(x, limite):
    """Devuelve [NB, NS, ZE, PS, PB]."""
    x = min(max(x, -limite), limite)
    return [
        left_shoulder(x, -limite, -0.35 * limite),
        trimf(x, -0.75 * limite, -0.30 * limite, 0.0),
        trimf(x, -0.20 * limite, 0.0, 0.20 * limite),
        trimf(x, 0.0, 0.30 * limite, 0.75 * limite),
        right_shoulder(x, 0.35 * limite, limite),
    ]


def mf_salida(label, x):
    if label == 0:
        return left_shoulder(x, -20, -7)
    if label == 1:
        return trimf(x, -15, -7, 0)
    if label == 2:
        return trimf(x, -4, 0, 4)
    if label == 3:
        return trimf(x, 0, 7, 15)
    return right_shoulder(x, 7, 20)


_X_OUT = np.arange(-20, 21, dtype=float)
_MF_OUT = np.array([[mf_salida(k, x) for x in _X_OUT] for k in range(5)])


class ControladorFuzzy:
    """Fuzzy PI incremental: (E, dE) -> dPWM. Mamdani, min/max, centroide."""

    def __init__(self, ku=KU_DEFECTO):
        self.ku = ku

    def _inferir(self, e, de):
        """
        Núcleo Mamdani compartido por calcular() y diagnostico().

        Devuelve (mu_e, mu_de, act, agregado, num, den):
            mu_e, mu_de : grados de pertenencia de las entradas  [NB..PB]
            act         : activación agregada por etiqueta de salida
            agregado    : conjunto de salida truncado, evaluado en _X_OUT
            num, den    : numerador y denominador del centroide
        """
        mu_e = np.asarray(membresias(e, E_MAX), dtype=float)
        mu_de = np.asarray(membresias(de, DE_MAX), dtype=float)

        act = np.zeros(5)

        for i in range(5):
            ei = mu_e[i]
            fila = REGLAS[i]
            for j in range(5):
                dj = mu_de[j]
                a = ei if ei < dj else dj          # AND = min
                s = fila[j]
                if a > act[s]:                     # agregación = max
                    act[s] = a

        agregado = np.max(np.minimum(act[:, None], _MF_OUT), axis=0)
        den = float(agregado.sum())
        num = float((_X_OUT * agregado).sum())

        return mu_e, mu_de, act, agregado, num, den

    def calcular(self, e, de):
        _, _, _, _, num, den = self._inferir(e, de)

        if den <= 1e-6:
            return 0.0

        return self.ku * (num / den)

    def diagnostico(self, e, de):
        """Como calcular(), pero además devuelve TODO el paso a paso fuzzy."""
        mu_e, mu_de, act, agregado, num, den = self._inferir(e, de)
        dpwm = 0.0 if den <= 1e-6 else self.ku * (num / den)

        i_dom = int(np.argmax(mu_e))
        j_dom = int(np.argmax(mu_de))

        # Regla más fuerte activada: la de mayor min(mu_e[i], mu_de[j]).
        # Con AND=min y agregación=max esa es la que manda la agregación.
        fuerza = -1.0
        regla = (i_dom, j_dom)

        for i in range(5):
            for j in range(5):
                a = min(mu_e[i], mu_de[j])
                if a > fuerza:
                    fuerza = a
                    regla = (i, j)

        return {
            "activo": True,
            "e": float(e),
            "de": float(de),
            "mu_e": mu_e,
            "mu_de": mu_de,
            "act": act,
            "agregado": agregado,
            "dpwm": float(dpwm),
            "e_dom": i_dom,
            "de_dom": j_dom,
            "sal_dom": int(np.argmax(act)),
            "regla": regla,
            "regla_out": REGLAS[regla[0]][regla[1]],
            "fuerza": float(fuerza),
        }


def diagnostico_inactivo():
    """El firmware no corre el fuzzy si SP < 1 RPM: todo a cero."""
    return {
        "activo": False,
        "e": 0.0,
        "de": 0.0,
        "mu_e": np.zeros(5),
        "mu_de": np.zeros(5),
        "act": np.zeros(5),
        "agregado": np.zeros(_X_OUT.size),
        "dpwm": 0.0,
        "e_dom": -1,
        "de_dom": -1,
        "sal_dom": -1,
        "regla": (-1, -1),
        "regla_out": -1,
        "fuerza": 0.0,
    }


class SimuladorMotor:
    """
    Motor DC:  tau * dw/dt = K * max(0, PWM - zona_muerta) - carga - w
    Mide RPM igual que el firmware (pulsos en ventana de N_VENTANA muestras).
    """

    def __init__(self):
        self.K = 0.6               # [RPM / unidad PWM]
        self.tau = 0.25            # [s]
        self.zona_muerta = 40.0    # [PWM]
        self.carga = 0.0           # [RPM] perturbación
        self.cuantizar = True
        self.ctrl = ControladorFuzzy()
        self.reset()

    def reset(self):
        self.t = 0.0
        self.w = 0.0
        self.sp = 0.0
        self.pwm = 0.0
        self.err_ant = 0.0
        self.acc_pulsos = 0.0
        self.ventana = deque(maxlen=N_VENTANA)

    def paso(self):
        # --- planta ---
        u = int(self.pwm)
        u_ef = max(0.0, u - self.zona_muerta)
        w_ss = max(0.0, self.K * u_ef - self.carga)
        alfa = 1.0 - math.exp(-TS / self.tau)
        self.w += alfa * (w_ss - self.w)

        # --- encoder ---
        pulsos = self.w / 60.0 * PULSOS_POR_VUELTA * TS
        if self.cuantizar:
            self.acc_pulsos += pulsos
            p = int(self.acc_pulsos)
            self.acc_pulsos -= p
        else:
            p = pulsos

        self.ventana.append((p, TS))
        suma_p = sum(v[0] for v in self.ventana)
        suma_t = sum(v[1] for v in self.ventana)
        rpm = (suma_p / PULSOS_POR_VUELTA) * (60.0 / suma_t)
        rpm = min(max(rpm, 0.0), RPM_MAX * 1.5)

        # --- control ---
        err = self.sp - rpm
        de = err - self.err_ant

        if self.sp < 1.0:
            dpwm = 0.0
            self.pwm = 0.0
            diag = diagnostico_inactivo()
        else:
            diag = self.ctrl.diagnostico(err, de)
            dpwm = diag["dpwm"]
            self.pwm = min(max(self.pwm + dpwm, 0.0), 255.0)

        self.err_ant = err
        self.t += TS

        return self.t, self.sp, rpm, self.pwm, err, dpwm, de, diag


def _integral(y, t):
    if len(t) < 2:
        return 0.0
    return float(np.sum((y[1:] + y[:-1]) * np.diff(t)) / 2.0)


def calcular_metricas(t, y, sp, y0):
    """
    Métricas de un escalón: t [s], y [RPM], sp = referencia, y0 = valor inicial.
    Devuelve dict con Mp [%], ts [s] (None si no ha establecido),
    ess [RPM] (None si hay poco tiempo), IAE, ITAE.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    t_rel = t - t[0]
    e = sp - y

    if sp >= y0:
        mp = max(0.0, (y.max() - sp) / sp * 100.0)
    else:
        mp = max(0.0, (sp - y.min()) / sp * 100.0)

    banda = max(0.02 * sp, 1.0)
    fuera = np.abs(e) > banda

    if not fuera.any():
        ts = 0.0
    else:
        ult = int(np.where(fuera)[0][-1])
        ts = None if ult == len(t_rel) - 1 else float(t_rel[ult + 1])

    ess = None
    if t_rel[-1] >= 2.0:
        m = t_rel >= (t_rel[-1] - 1.0)
        ess = float(np.mean(e[m]))

    return {
        "mp": float(mp),
        "ts": ts,
        "ess": ess,
        "iae": _integral(np.abs(e), t_rel),
        "itae": _integral(t_rel * np.abs(e), t_rel),
    }


def generar_figuras():
    """PNG de membresías y superficie de control para el artículo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    etiquetas = ["NB", "NS", "ZE", "PS", "PB"]

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))

    xs = np.linspace(-E_MAX, E_MAX, 401)
    for k in range(5):
        ax[0].plot(xs, [membresias(x, E_MAX)[k] for x in xs], label=etiquetas[k])
    ax[0].set_title("Entrada: error e [RPM]")

    xs = np.linspace(-DE_MAX, DE_MAX, 401)
    for k in range(5):
        ax[1].plot(xs, [membresias(x, DE_MAX)[k] for x in xs], label=etiquetas[k])
    ax[1].set_title("Entrada: variación de error Δe [RPM/muestra]")

    xs = np.linspace(-20, 20, 401)
    for k in range(5):
        ax[2].plot(xs, [mf_salida(k, x) for x in xs], label=etiquetas[k])
    ax[2].set_title("Salida: ΔPWM")

    for a in ax:
        a.set_ylabel("Grado de pertenencia")
        a.grid(True, alpha=0.3)
        a.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig("fig_membresias.png", dpi=300)

    ctrl = ControladorFuzzy(ku=1.0)
    e_v = np.linspace(-E_MAX, E_MAX, 41)
    d_v = np.linspace(-DE_MAX, DE_MAX, 41)
    Z = np.array([[ctrl.calcular(e, d) for e in e_v] for d in d_v])
    E, D = np.meshgrid(e_v, d_v)

    fig2 = plt.figure(figsize=(6, 4.5))
    ax3 = fig2.add_subplot(111, projection="3d")
    ax3.plot_surface(E, D, Z, cmap="viridis")
    ax3.set_xlabel("e [RPM]")
    ax3.set_ylabel("Δe [RPM/muestra]")
    ax3.set_zlabel("ΔPWM")
    ax3.set_title("Superficie de control")
    fig2.tight_layout()
    fig2.savefig("fig_superficie.png", dpi=300)

    print("Generadas: fig_membresias.png, fig_superficie.png")


# ======================================================================
# INTERFAZ
# ======================================================================

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QSlider,
    QPushButton,
    QComboBox,
    QGroupBox,
    QDoubleSpinBox,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QTabWidget,
    QSplitter,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QHeaderView,
)

import pyqtgraph as pg

# Curvas de membresía de entrada, precalculadas para las gráficas estáticas
# (cada fila es un punto del universo; columna k = etiqueta k)
_XS_E = np.linspace(-E_MAX, E_MAX, 401)
_MU_E = np.array([membresias(x, E_MAX) for x in _XS_E]).T
_XS_DE = np.linspace(-DE_MAX, DE_MAX, 401)
_MU_DE = np.array([membresias(x, DE_MAX) for x in _XS_DE]).T


def brocha(color_hex, alpha):
    c = QColor(color_hex)
    c.setAlpha(alpha)
    return pg.mkBrush(c)


class BarrasFuzzy(QWidget):
    """
    Las 5 barras de grado de pertenencia NB/NS/ZE/PS/PB en vivo.
    La etiqueta dominante (mayor μ) se resalta.
    """

    def __init__(self, titulo, padre=None):
        super().__init__(padre)
        self.titulo = titulo
        self.valores = [0.0] * 5
        self.dominante = -1
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_valores(self, valores, dominante=-1):
        self.valores = [min(max(float(v), 0.0), 1.0) for v in valores]
        self.dominante = dominante
        self.update()

    def limpiar(self):
        self.set_valores([0.0] * 5, -1)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#161a20"))

        f = QFont(self.font())
        f.setPointSize(10)
        f.setBold(True)
        p.setFont(f)

        lineas = self.titulo.split("\n")

        for n, linea in enumerate(lineas):
            p.setPen(QPen(QColor("#e0e0e0" if n == 0 else "#9aa0a6")))
            p.drawText(6, 17 + 16 * n, linea)

        w = self.width()
        alto = self.height()
        top = 18 + 16 * len(lineas)
        pie = alto - 24
        util = pie - top
        ancho = (w - 12) / 5.0

        for k in range(5):
            x0 = 6 + k * ancho
            # fondo de la celda
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#20252d"))
            p.drawRect(QRectF(x0 + 2, top, ancho - 4, util))

            v = self.valores[k]
            # barra
            if v > 0.0:
                p.setBrush(QColor(COLORES_ET[k]))
                p.drawRect(QRectF(x0 + 2, pie - util * v, ancho - 4, util * v))

            # etiqueta
            f.setPointSize(13)
            f.setBold(k == self.dominante)
            p.setFont(f)
            p.setPen(QPen(QColor(COLORES_ET[k] if k == self.dominante else "#9aa0a6")))
            p.drawText(
                QRectF(x0, pie + 4, ancho, 20),
                Qt.AlignCenter,
                ETIQUETAS[k],
            )
            p.drawText(
                QRectF(x0, pie + 4, ancho - 4, 20),
                Qt.AlignRight | Qt.AlignVCenter,
                f"{v:.3f}",
            )

            if k == self.dominante:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor("#ffffff"), 2))
                p.drawRect(QRectF(x0 + 2, top, ancho - 4, util))


class TablaMembresias(QTableWidget):
    """
    Tabla con el grado de pertenencia de cada restricción (NB NS ZE PS PB)
    para e, Δe y la salida, más la etiqueta que manda en cada fila.
    El color del fondo es la intensidad del grado y la celda dominante
    queda marcada en negrita con borde.
    """

    FILAS = ["μ ERROR  (e)", "μ ΔERROR  (Δe)", "μ SALIDA  (agregada)"]

    def __init__(self, padre=None):
        super().__init__(padre)
        self.setColumnCount(len(ETIQUETAS) + 2)
        self.setRowCount(len(self.FILAS))

        self.setHorizontalHeaderLabels(["restricción"] + ETIQUETAS + ["→ manda"])
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        self.setMinimumWidth(470)
        self.setMinimumHeight(190)

        for i, nombre in enumerate(self.FILAS):
            c = QTableWidgetItem(nombre)
            c.setTextAlignment(Qt.AlignCenter)
            f = QFont(self.font())
            f.setPointSize(10)
            f.setBold(True)
            c.setFont(f)
            c.setForeground(QBrush(QColor("#e0e0e0")))
            self.setItem(i, 0, c)

        for i in range(len(self.FILAS)):
            for k in range(len(ETIQUETAS)):
                c = QTableWidgetItem("0.000")
                c.setTextAlignment(Qt.AlignCenter)
                f = QFont("Consolas")
                f.setPointSize(12)
                f.setBold(True)
                c.setFont(f)
                self.setItem(i, k + 1, c)

            c = QTableWidgetItem("—")
            c.setTextAlignment(Qt.AlignCenter)
            f = QFont(self.font())
            f.setPointSize(12)
            f.setBold(True)
            c.setFont(f)
            self.setItem(i, len(ETIQUETAS) + 1, c)

        self.limpiar()

    def limpiar(self):
        self.set_datos(
            [np.zeros(5), np.zeros(5), np.zeros(5)], [-1, -1, -1], activo=False
        )

    def set_datos(self, filas, dominantes, activo=True):
        for i, (vals, dom) in enumerate(zip(filas, dominantes)):
            for k, v in enumerate(vals):
                c = self.item(i, k + 1)
                v = 0.0 if not activo else min(max(float(v), 0.0), 1.0)
                c.setText("—" if not activo else f"{v:.3f}")

                fondo = QColor(COLORES_ET[k])
                fondo.setAlpha(int(30 + 205 * v))
                c.setBackground(QBrush(fondo))

                negrita = activo and k == dom
                f = QFont("Consolas")
                f.setPointSize(12)
                f.setBold(negrita)
                c.setFont(f)
                c.setForeground(
                    QBrush(QColor("#ffffff" if v > 0.5 else "#e8eaed"))
                )

            dc = self.item(i, len(ETIQUETAS) + 1)
            dc.setText("—" if not activo else ETIQUETAS[dom])
            dc.setForeground(
                QBrush(QColor("#9aa0a6" if dom < 0 else COLORES_ET[dom]))
            )


class MapaReglas(QWidget):
    """
    Base de reglas 5x5: intensidad = min(mu_e[i], mu_de[j]) (lo que dispara
    esa regla) y el texto es la etiqueta de salida a la que apunta.
    La celda de la regla más fuerte queda marcada.
    """

    def __init__(self, padre=None):
        super().__init__(padre)
        self.mu_e = [0.0] * 5
        self.mu_de = [0.0] * 5
        self.fuerte = (-1, -1)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_estado(self, mu_e, mu_de, regla):
        self.mu_e = list(mu_e)
        self.mu_de = list(mu_de)
        self.fuerte = tuple(regla)
        self.update()

    def limpiar(self):
        self.set_estado([0.0] * 5, [0.0] * 5, (-1, -1))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#161a20"))

        f = QFont(self.font())
        f.setPointSize(10)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QPen(QColor("#e0e0e0")))
        p.drawText(6, 17, "Base de reglas  (filas e / columnas Δe)")

        margen = 28
        w = self.width()
        alto = self.height()
        top = 24
        ancho = (w - margen - 6) / 5.0
        alto_celda = (alto - top - 18) / 5.0

        for k in range(5):
            p.setPen(QPen(QColor("#9aa0a6")))
            p.drawText(
                QRectF(0, top + k * alto_celda, margen - 4, alto_celda),
                Qt.AlignCenter,
                ETIQUETAS[k],
            )
            p.drawText(
                QRectF(margen + k * ancho, alto - 16, ancho, 16),
                Qt.AlignCenter,
                ETIQUETAS[k],
            )

        for i in range(5):
            for j in range(5):
                a = min(self.mu_e[i], self.mu_de[j])
                x0 = margen + j * ancho
                y0 = top + i * alto_celda

                c = QColor(COLORES_ET[REGLAS[i][j]])
                c.setAlpha(int(40 + 200 * min(max(a, 0.0), 1.0)))
                p.setPen(Qt.NoPen)
                p.setBrush(c)
                p.drawRect(QRectF(x0 + 1, y0 + 1, ancho - 2, alto_celda - 2))

                if (i, j) == self.fuerte and a > 0.0:
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(QColor("#ffffff"), 3))
                    p.drawRect(QRectF(x0 + 1, y0 + 1, ancho - 2, alto_celda - 2))

                p.setFont(f)
                p.setPen(QPen(QColor("#ffffff") if a > 0.45 else "#c8ccd0"))
                p.drawText(
                    QRectF(x0, y0, ancho, alto_celda),
                    Qt.AlignCenter,
                    ETIQUETAS[REGLAS[i][j]],
                )


# ======================================================================
# ENLACE SERIAL ROBUSTO
#
# Problemas que solve (causan "GUI pesada" + "el motor no arranca"):
#
#  1) readline() BLOQUEA. Con Serial(timeout=0.01) cada readline() espera
#     hasta 10 ms si la linea todavia no llego completa. Con hasta 200
#     iteraciones por tick eso son 2 s de interfaz CONGELADA. Ademas el
#     ESP32 imprime varios kB de boot log al reiniciarse y la primer
#     linea siempre llega partida -> readline() se queda esperando.
#     Aqui se usa timeout=0 (no bloqueante) + buffer propio.
#
#  2) DTR/RTS reinician el ESP32. Al abrir el puerto pyserial pone
#     DTR/RTS a 1 -> el ESP32 se reinicia. En placas con USB nativo
#     (ESP32-S3 / C3) ademas cada escritura puede reiniciar el board,
#     con lo que el setpoint se pierde justo despues de enviarlo y el
#     motor nunca arranca.
#
#  3) Los bytes enviados antes de que termine el arranque del ESP32 se
#     descartan en silencio. Por eso ahora el HMI espera el banner
#     "FUZZY_MOTOR_READY" y reenvia KU= y SP= al detectarlo.
# ======================================================================


class EnlaceESP32:
    """Lector/escritor NO bloqueante sobre pyserial."""

    BANNER = "FUZZY_MOTOR_READY"

    def __init__(self, puerto, baudrate=115200):
        # timeout=0  ->  read()/write() nunca bloquean el hilo de la GUI
        self.ser = serial.Serial(
            port=puerto, baudrate=baudrate, timeout=0, write_timeout=0.5
        )

        # Desactiva el reinicio automatico por DTR/RTS
        try:
            self.ser.dtr = False
            self.ser.rts = False
        except Exception:
            pass

        self.buf = bytearray()
        self.lineas_ok = 0
        self.lineas_malas = 0
        self.escritas = 0
        self.error = None
        self.fallos = 0
        self.ultimo_dato = time.perf_counter()

    # -- lectura: nunca bloquea -------------------------------------------
    def leer(self):
        """Devuelve la lista de lineas completas. No bloquea NUNCA."""
        lineas = []

        try:
            n = self.ser.in_waiting
            if n:
                self.buf.extend(self.ser.read(n))

            while True:
                i = self.buf.find(b"\n")
                if i < 0:
                    break
                cruda = bytes(self.buf[:i])
                del self.buf[: i + 1]
                s = cruda.decode("utf-8", errors="ignore").strip()
                if s:
                    lineas.append(s)

            # Si se acumulan bytes sin salto de linea (basura / boot log),
            # se descarta lo viejo para que el buffer no crezca sin fin.
            if len(self.buf) > 4096:
                del self.buf[:-1024]

            self.fallos = 0
            self.error = None

        except Exception as e:
            self.error = e
            self.fallos += 1

        if lineas:
            self.ultimo_dato = time.perf_counter()
            self.lineas_ok += len(lineas)

        return lineas

    # -- escritura --------------------------------------------------------
    def escribir(self, texto):
        try:
            self.ser.write((texto + "\n").encode())
            self.escritas += 1
            self.error = None
            return True
        except Exception as e:
            self.error = e
            return False

    def sin_datos_desde(self):
        return time.perf_counter() - self.ultimo_dato

    def cerrar(self):
        try:
            self.ser.close()
        except Exception:
            pass


class HMI(QWidget):

    VELOCIDADES = [1, 2, 5, 10, 20]

    def __init__(self):
        super().__init__()

        self.setWindowTitle("HMI - Control Fuzzy Motor DC (Simulación / Real)")
        self.resize(1420, 960)

        self.modo = "sim"
        self.enlace = None
        self.t0_real = None

        # handshake / robustez del enlace
        self.esp32_listo = False
        self.sp_deseado = 0.0
        self._t_ultimo = None
        self._t_envio_sp = 0.0
        self._t_ping = 0.0
        self._lineas_malas = 0

        self.sim = SimuladorMotor()
        self.sim_activa = False
        self._ult_tick = None
        self._acum = 0.0
        self.velocidad = 1

        # Controlador solo para visualizar la inferencia en modo real
        self.fz_view = ControladorFuzzy(ku=KU_DEFECTO)

        self.ventana = 30.0

        self.limpiar_datos()
        self.build_ui()

        self.aplicar_params_sim()
        self.detectar_puertos()
        self.cambiar_modo(0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(20)

        self.timer_met = QTimer(self)
        self.timer_met.timeout.connect(self.actualizar_metricas)
        self.timer_met.start(500)

        self.timer_estado = QTimer(self)
        self.timer_estado.timeout.connect(self.watchdog_enlace)
        self.timer_estado.start(1000)

    # ------------------------------------------------------------------
    # Utilidades de UI
    # ------------------------------------------------------------------

    def spin(self, minimo, maximo, valor, paso, decimales=2):
        s = QDoubleSpinBox()
        s.setRange(minimo, maximo)
        s.setDecimals(decimales)
        s.setSingleStep(paso)
        s.setValue(valor)
        return s

    def crear_metrica(self, titulo, valor, tam=24):
        caja = QGroupBox(titulo)
        lay = QVBoxLayout(caja)
        label = QLabel(valor)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(f"font-size:{tam}px; font-weight:bold;")
        lay.addWidget(label)
        caja.valor = label
        return caja

    def crear_plot(self, titulo, ylabel):
        p = pg.PlotWidget(title=titulo)
        p.showGrid(x=True, y=True)
        p.setLabel("left", ylabel)
        p.setLabel("bottom", "Tiempo", units="s")
        return p

    def build_ui(self):
        root = QVBoxLayout(self)

        # ---------------- Modo ----------------
        gb_modo = QGroupBox("Modo de operación")
        h = QHBoxLayout(gb_modo)
        h.addWidget(QLabel("Modo:"))
        self.combo_modo = QComboBox()
        self.combo_modo.addItems(["Simulación", "Real (ESP32)"])
        h.addWidget(self.combo_modo)
        h.addStretch()
        root.addWidget(gb_modo)
        self.combo_modo.currentIndexChanged.connect(self.cambiar_modo)

        # ---------------- Parámetros de simulación ----------------
        self.gb_sim = QGroupBox("Simulación del motor DC (1er orden + zona muerta)")
        h = QHBoxLayout(self.gb_sim)

        self.sp_K = self.spin(0.05, 5.0, 0.6, 0.05, 2)
        self.sp_tau = self.spin(0.02, 5.0, 0.25, 0.05, 2)
        self.sp_zm = self.spin(0, 120, 40, 5, 0)
        self.sp_carga = self.spin(0, 80, 0, 5, 1)
        self.chk_cuant = QCheckBox("Cuantización encoder")
        self.chk_cuant.setChecked(True)
        self.combo_vel = QComboBox()
        self.combo_vel.addItems([f"{v}x" for v in self.VELOCIDADES])
        self.btn_sim = QPushButton("▶ Iniciar")
        self.btn_sim_reset = QPushButton("Reiniciar")

        for texto, w in [
            ("K [RPM/PWM]", self.sp_K),
            ("τ [s]", self.sp_tau),
            ("Zona muerta [PWM]", self.sp_zm),
            ("Carga [RPM]", self.sp_carga),
        ]:
            h.addWidget(QLabel(texto))
            h.addWidget(w)

        h.addWidget(self.chk_cuant)
        h.addWidget(QLabel("Velocidad"))
        h.addWidget(self.combo_vel)
        h.addWidget(self.btn_sim)
        h.addWidget(self.btn_sim_reset)
        root.addWidget(self.gb_sim)

        for w in (self.sp_K, self.sp_tau, self.sp_zm, self.sp_carga):
            w.valueChanged.connect(self.aplicar_params_sim)
        self.chk_cuant.toggled.connect(self.aplicar_params_sim)
        self.combo_vel.currentIndexChanged.connect(self.cambiar_velocidad)
        self.btn_sim.clicked.connect(self.toggle_sim)
        self.btn_sim_reset.clicked.connect(self.reiniciar_sim)

        # ---------------- Conexión real ----------------
        self.gb_real = QGroupBox("Comunicación ESP32")
        h = QHBoxLayout(self.gb_real)
        self.combo_puerto = QComboBox()
        self.combo_puerto.setMinimumWidth(260)
        self._puertos = {}
        b_act = QPushButton("Actualizar puertos")
        b_con = QPushButton("Conectar")
        b_des = QPushButton("Desconectar")
        self.estado = QLabel("DESCONECTADO")
        self.estado.setAlignment(Qt.AlignCenter)

        h.addWidget(QLabel("Puerto:"))
        h.addWidget(self.combo_puerto)
        h.addWidget(b_act)
        h.addWidget(b_con)
        h.addWidget(b_des)
        h.addWidget(self.estado)
        root.addWidget(self.gb_real)

        self.detalle = QLabel("")
        self.detalle.setAlignment(Qt.AlignCenter)
        self.detalle.setStyleSheet("color:#666; font-size:11px;")
        root.addWidget(self.detalle)

        b_act.clicked.connect(self.detectar_puertos)
        b_con.clicked.connect(self.conectar)
        b_des.clicked.connect(lambda: self.desconectar())

        # ---------------- Indicadores ----------------
        fila = QHBoxLayout()
        self.lbl_sp = self.crear_metrica("SETPOINT", "0 RPM")
        self.lbl_rpm = self.crear_metrica("RPM REAL", "0 RPM")
        self.lbl_error = self.crear_metrica("ERROR", "0 RPM")
        self.lbl_pwm = self.crear_metrica("PWM", "0")
        for c in (self.lbl_sp, self.lbl_rpm, self.lbl_error, self.lbl_pwm):
            fila.addWidget(c)
        root.addLayout(fila)

        # ---------------- Métricas del escalón ----------------
        fila = QHBoxLayout()
        self.m_mp = self.crear_metrica("Sobreimpulso", "—", 18)
        self.m_ts = self.crear_metrica("t establec. (2%)", "—", 18)
        self.m_ess = self.crear_metrica("Error estable", "—", 18)
        self.m_iae = self.crear_metrica("IAE", "—", 18)
        self.m_itae = self.crear_metrica("ITAE", "—", 18)
        for c in (self.m_mp, self.m_ts, self.m_ess, self.m_iae, self.m_itae):
            fila.addWidget(c)
        root.addLayout(fila)

        # ---------------- Control ----------------
        gb_ctrl = QGroupBox("Referencia y ajustes")
        g = QGridLayout(gb_ctrl)

        g.addWidget(QLabel("Setpoint [RPM]"), 0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        g.addWidget(self.slider, 0, 1)
        self.lbl_slider = QLabel("50 RPM")
        g.addWidget(self.lbl_slider, 0, 2)

        b_env = QPushButton("Enviar setpoint")
        b_stop = QPushButton("STOP")
        g.addWidget(b_env, 0, 3)
        g.addWidget(b_stop, 0, 4)

        g.addWidget(QLabel("KU (ganancia salida)"), 1, 0)
        self.sp_ku = self.spin(0.05, 3.0, KU_DEFECTO, 0.05, 2)
        g.addWidget(self.sp_ku, 1, 1)

        g.addWidget(QLabel("Ventana [s]"), 1, 2)
        self.sp_ventana = self.spin(5, 120, 30, 5, 0)
        g.addWidget(self.sp_ventana, 1, 3)

        b_csv = QPushButton("Guardar CSV")
        b_lim = QPushButton("Limpiar gráficas")
        g.addWidget(b_csv, 1, 4)
        g.addWidget(b_lim, 1, 5)
        root.addWidget(gb_ctrl)

        self.slider.valueChanged.connect(
            lambda v: self.lbl_slider.setText(f"{v} RPM")
        )
        b_env.clicked.connect(self.enviar_setpoint)
        b_stop.clicked.connect(self.enviar_stop)
        self.sp_ku.valueChanged.connect(self.cambiar_ku)
        self.sp_ventana.valueChanged.connect(
            lambda v: setattr(self, "ventana", float(v))
        )
        b_csv.clicked.connect(self.guardar_csv)
        b_lim.clicked.connect(self.limpiar)

        # ---------------- Gráficas ----------------
        self.tabs = QTabWidget()

        pag_ctrl = QWidget()
        grid = QGridLayout(pag_ctrl)

        self.plot_rpm = self.crear_plot("Velocidad", "RPM")
        self.plot_rpm.addLegend(offset=(10, 10))
        self.curva_sp = self.plot_rpm.plot(
            pen=pg.mkPen("r", width=2, style=Qt.DashLine), name="Setpoint"
        )
        self.curva_rpm = self.plot_rpm.plot(
            pen=pg.mkPen("c", width=2), name="RPM real"
        )

        self.plot_error = self.crear_plot("Error", "Error [RPM]")
        self.plot_error.addLine(y=0, pen=pg.mkPen("w", style=Qt.DotLine))
        self.curva_error = self.plot_error.plot(pen=pg.mkPen("y", width=2))

        self.plot_pwm = self.crear_plot("PWM", "PWM")
        self.plot_pwm.setYRange(0, 260)
        self.curva_pwm = self.plot_pwm.plot(pen=pg.mkPen("g", width=2))

        self.plot_dpwm = self.crear_plot("Salida del fuzzy (ΔPWM)", "ΔPWM")
        self.plot_dpwm.addLine(y=0, pen=pg.mkPen("w", style=Qt.DotLine))
        self.curva_dpwm = self.plot_dpwm.plot(pen=pg.mkPen("m", width=2))

        for p in (self.plot_error, self.plot_pwm, self.plot_dpwm):
            p.setXLink(self.plot_rpm)

        grid.addWidget(self.plot_rpm, 0, 0)
        grid.addWidget(self.plot_error, 0, 1)
        grid.addWidget(self.plot_pwm, 1, 0)
        grid.addWidget(self.plot_dpwm, 1, 1)
        self.tabs.addTab(pag_ctrl, "Control")
        self.tabs.addTab(self.construir_pestana_valores(), "Fuzzy · valores (NB NS ZE PS PB)")
        self.tabs.addTab(self.construir_pestana_entradas(), "Fuzzy · entradas (e, Δe)")
        self.tabs.addTab(self.construir_pestana_salida(), "Fuzzy · salida y reglas")
        self.tabs.addTab(self.construir_pestana_hist(), "Fuzzy · μ en el tiempo")
        root.addWidget(self.tabs, 1)

    def construir_pestana_valores(self):
        """
        Pestaña del fuzzy 1: SOLO los valores de las restricciones.
        Barras grandes + tabla numérica + resumen del paso.
        """
        pag = QWidget()
        raiz = QVBoxLayout(pag)
        raiz.setContentsMargins(6, 6, 6, 6)

        arriba = QSplitter(Qt.Horizontal)

        barras = QWidget()
        lb = QHBoxLayout(barras)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.setSpacing(8)
        self.barras_e = BarrasFuzzy("μ ERROR\n\n  e = SP - RPM")
        self.barras_de = BarrasFuzzy("μ ΔERROR\n\n  (RPM / muestra)")
        self.barras_sal = BarrasFuzzy("μ SALIDA\n\n  activación agregada")
        for w in (self.barras_e, self.barras_de, self.barras_sal):
            w.setMinimumHeight(300)
            lb.addWidget(w)

        self.tabla_mu = TablaMembresias()
        self.tabla_mu.setMinimumHeight(300)

        arriba.addWidget(barras)
        arriba.addWidget(self.tabla_mu)
        arriba.setStretchFactor(0, 3)
        arriba.setStretchFactor(1, 2)
        arriba.setSizes([820, 560])

        # --- resumen del paso fuzzy ---
        self.lbl_fuzzy = QLabel("sin datos")
        self.lbl_fuzzy.setWordWrap(True)
        self.lbl_fuzzy.setAlignment(Qt.AlignCenter)
        self.lbl_fuzzy.setStyleSheet(
            "font-size:14px; padding:10px; background:#1b2027; border-radius:4px;"
        )
        self.lbl_fuzzy.setMinimumHeight(80)

        vertical = QSplitter(Qt.Vertical)
        vertical.addWidget(arriba)
        vertical.addWidget(self.lbl_fuzzy)
        vertical.setStretchFactor(0, 1)
        vertical.setStretchFactor(1, 0)
        vertical.setSizes([520, 100])

        raiz.addWidget(vertical)
        return pag

    def construir_pestana_entradas(self):
        """
        Pestaña del fuzzy 2: fuzzificación de las ENTRADAS.
        Las 5 funciones de pertenencia de e y de Δe con el cursor en el
        valor actual -> se ve qué restricción (NB NS ZE PS PB) está activa.
        """
        pag = QWidget()

        # --- funciones de pertenencia con cursor en el valor actual ---
        self.plot_mf_e = self.crear_plot("Funciones de pertenencia del ERROR", "μ")
        self.plot_mf_e.setLabel("bottom", "e [RPM]")
        self.plot_mf_e.setXRange(-E_MAX, E_MAX, padding=0.02)
        self.plot_mf_e.setYRange(0, 1.12, padding=0.02)
        for k in range(5):
            self.plot_mf_e.plot(_XS_E, _MU_E[k], pen=pg.mkPen(COLORES_ET[k], width=2))
        self.cursor_e = pg.InfiniteLine(angle=90, pen=pg.mkPen("w", width=2))
        self.plot_mf_e.addItem(self.cursor_e)

        self.plot_mf_de = self.crear_plot("Funciones de pertenencia de Δe", "μ")
        self.plot_mf_de.setLabel("bottom", "Δe [RPM/muestra]")
        self.plot_mf_de.setXRange(-DE_MAX, DE_MAX, padding=0.02)
        self.plot_mf_de.setYRange(0, 1.12, padding=0.02)
        for k in range(5):
            self.plot_mf_de.plot(
                _XS_DE, _MU_DE[k], pen=pg.mkPen(COLORES_ET[k], width=2)
            )
        self.cursor_de = pg.InfiniteLine(angle=90, pen=pg.mkPen("w", width=2))
        self.plot_mf_de.addItem(self.cursor_de)

        g = QGridLayout(pag)
        g.setContentsMargins(4, 4, 4, 4)

        for w in (self.plot_mf_e, self.plot_mf_de):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            g.addWidget(w, 0, g.count())

        g.setColumnStretch(0, 1)
        g.setColumnStretch(1, 1)

        return pag

    def construir_pestana_salida(self):
        """
        Pestaña del fuzzy 3: defuzzificación y base de reglas.
        Conjunto de salida agregado con su centroide, y la base 5x5 con
        la intensidad de cada regla (min(mu_e[i], mu_de[j])).
        """
        pag = QWidget()

        # --- defuzzificación: conjunto agregado + centroide ---
        self.plot_sal = self.crear_plot(
            "Conjunto de SALIDA agregado y centroide (defuzzificación)", "μ"
        )
        self.plot_sal.setLabel("bottom", "ΔPWM")
        self.plot_sal.setXRange(-21, 21, padding=0.02)
        self.plot_sal.setYRange(0, 1.12, padding=0.02)
        for k in range(5):
            self.plot_sal.plot(_X_OUT, _MF_OUT[k], pen=pg.mkPen(COLORES_ET[k], width=1))
        self.curva_agregado = self.plot_sal.plot(
            _X_OUT,
            np.zeros(_X_OUT.size),
            pen=pg.mkPen("#eceff1", width=2),
            fillLevel=0.0,
            brush=brocha("#cfd8dc", 90),
        )
        self.marca_centro = self.plot_sal.plot(
            [0],
            [0],
            pen=None,
            symbol="o",
            symbolBrush="w",
            symbolPen="k",
            symbolSize=12,
        )
        self.cursor_sal = pg.InfiniteLine(
            angle=90, pen=pg.mkPen("w", width=1, style=Qt.DashLine)
        )
        self.plot_sal.addItem(self.cursor_sal)

        # --- base de reglas ---
        self.mapa = MapaReglas()

        g = QGridLayout(pag)
        g.setContentsMargins(4, 4, 4, 4)

        for w in (self.plot_sal, self.mapa):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            g.addWidget(w, 0, g.count())

        g.setColumnStretch(0, 1)
        g.setColumnStretch(1, 1)

        return pag

    def construir_pestana_hist(self):
        """Pestaña 2 del fuzzy: los mismos grados de pertenencia vs tiempo."""
        pag = QWidget()
        raiz = QVBoxLayout(pag)
        raiz.setContentsMargins(4, 4, 4, 4)

        self.plot_mu_e = self.crear_plot(
            "Grado de pertenencia del ERROR  (μ NB…PB)", "μ"
        )
        self.plot_mu_e.addLegend(offset=(8, 8))
        self.plot_mu_e.setYRange(0, 1.12, padding=0.02)
        self.curvas_mu_e = self.crear_curvas_mu(self.plot_mu_e)

        self.plot_mu_de = self.crear_plot(
            "Grado de pertenencia de ΔERROR  (μ NB…PB)", "μ"
        )
        self.plot_mu_de.addLegend(offset=(8, 8))
        self.plot_mu_de.setYRange(0, 1.12, padding=0.02)
        self.curvas_mu_de = self.crear_curvas_mu(self.plot_mu_de)

        self.plot_mu_sal = self.crear_plot(
            "Activación agregada de la SALIDA  (μ NB…PB)", "μ"
        )
        self.plot_mu_sal.addLegend(offset=(8, 8))
        self.plot_mu_sal.setYRange(0, 1.12, padding=0.02)
        self.curvas_mu_sal = self.crear_curvas_mu(self.plot_mu_sal)

        for p in (self.plot_mu_e, self.plot_mu_de, self.plot_mu_sal):
            p.setXLink(self.plot_rpm)
            p.setMinimumHeight(210)

        vertical = QSplitter(Qt.Vertical)
        for p in (self.plot_mu_e, self.plot_mu_de, self.plot_mu_sal):
            vertical.addWidget(p)
        vertical.setSizes([320, 320, 320])

        raiz.addWidget(vertical)
        return pag

    def crear_curvas_mu(self, plot):
        """Las 5 curvas μ_k(t) con relleno, una por etiqueta NB…PB."""
        return [
            plot.plot(
                pen=pg.mkPen(COLORES_ET[k], width=1.6),
                fillLevel=1.0,
                brush=brocha(COLORES_ET[k], 55),
                name=ETIQUETAS[k],
            )
            for k in range(5)
        ]

    # ------------------------------------------------------------------
    # Datos
    # ------------------------------------------------------------------

    def limpiar_datos(self):
        self.t = []
        self.sp = []
        self.rpm = []
        self.err = []
        self.pwm = []
        self.dpwm = []
        self.idx_esc = None
        self.y0 = 0.0
        self.sp_ultimo = None

        # traza fuzzy
        self.de = []
        self.mu_e = []
        self.mu_de = []
        self.mu_sal = []
        self.diag = None
        self.err_ant_hmi = 0.0

    def limpiar(self):
        self.limpiar_datos()
        self.t0_real = None
        for c in (
            self.curva_sp,
            self.curva_rpm,
            self.curva_error,
            self.curva_pwm,
            self.curva_dpwm,
        ):
            c.setData([], [])
        self.limpiar_fuzzy()
        self.lbl_sp.valor.setText("0 RPM")
        self.lbl_rpm.valor.setText("0 RPM")
        self.lbl_error.valor.setText("0 RPM")
        self.lbl_pwm.valor.setText("0")
        self.mostrar_metricas(None)

    def limpiar_fuzzy(self):
        for c in self.curvas_mu_e + self.curvas_mu_de + self.curvas_mu_sal:
            c.setData([], [])
        self.barras_e.limpiar()
        self.barras_de.limpiar()
        self.barras_sal.limpiar()
        self.tabla_mu.limpiar()
        self.mapa.limpiar()
        self.curva_agregado.setData(_X_OUT, np.zeros(_X_OUT.size))
        self.marca_centro.setData([0], [0])
        self.cursor_e.setValue(0)
        self.cursor_de.setValue(0)
        self.cursor_sal.setValue(0)
        self.lbl_fuzzy.setText("sin datos")

    def registrar(self, t, sp, rpm, pwm, err, dpwm, de=None, diag=None):
        # Detectar escalón de referencia
        if self.sp_ultimo is None or abs(sp - self.sp_ultimo) > 0.5:
            self.idx_esc = len(self.t)
            self.y0 = self.rpm[-1] if self.rpm else 0.0
        self.sp_ultimo = sp

        # En modo real el ESP32 manda ERR y DPWM, pero no Δe: se reconstruye
        # con la diferencia de errores recibidos y se vuelve a inferir con las
        # MISMAS tablas del firmware, así se ve qué etiquetas están activas.
        if de is None:
            de = err - self.err_ant_hmi

        if sp < 1.0:
            diag = diagnostico_inactivo()
        elif diag is None:
            self.fz_view.ku = self.sp_ku.value()
            diag = self.fz_view.diagnostico(err, de)

        self.err_ant_hmi = err
        self.diag = diag

        self.t.append(t)
        self.sp.append(sp)
        self.rpm.append(rpm)
        self.pwm.append(pwm)
        self.err.append(err)
        self.dpwm.append(dpwm)
        self.de.append(de)
        self.mu_e.append(np.asarray(diag["mu_e"], dtype=float))
        self.mu_de.append(np.asarray(diag["mu_de"], dtype=float))
        self.mu_sal.append(np.asarray(diag["act"], dtype=float))

    def refrescar(self):
        if not self.t:
            return

        t_now = self.t[-1]
        i0 = bisect_left(self.t, t_now - self.ventana)
        tt = self.t[i0:]

        self.curva_sp.setData(tt, self.sp[i0:])
        self.curva_rpm.setData(tt, self.rpm[i0:])
        self.curva_error.setData(tt, self.err[i0:])
        self.curva_pwm.setData(tt, self.pwm[i0:])
        self.curva_dpwm.setData(tt, self.dpwm[i0:])

        ventana_e = np.array(self.mu_e[i0:], dtype=float).reshape(-1, 5)
        ventana_de = np.array(self.mu_de[i0:], dtype=float).reshape(-1, 5)
        ventana_sal = np.array(self.mu_sal[i0:], dtype=float).reshape(-1, 5)

        for k in range(5):
            self.curvas_mu_e[k].setData(tt, ventana_e[:, k])
            self.curvas_mu_de[k].setData(tt, ventana_de[:, k])
            self.curvas_mu_sal[k].setData(tt, ventana_sal[:, k])

        self.plot_rpm.setXRange(
            max(0.0, t_now - self.ventana),
            max(self.ventana, t_now),
            padding=0,
        )

        self.lbl_sp.valor.setText(f"{self.sp[-1]:.1f} RPM")
        self.lbl_rpm.valor.setText(f"{self.rpm[-1]:.1f} RPM")
        self.lbl_error.valor.setText(f"{self.err[-1]:.1f} RPM")
        self.lbl_pwm.valor.setText(f"{self.pwm[-1]:.0f}")

        self.refrescar_fuzzy()

    # ------------------------------------------------------------------
    # Panel fuzzy: qué etiqueta (NB NS ZE PS PB) manda en cada paso
    # ------------------------------------------------------------------

    def refrescar_fuzzy(self):
        d = self.diag

        if d is None:
            return

        if not d["activo"]:
            self.barras_e.limpiar()
            self.barras_de.limpiar()
            self.barras_sal.limpiar()
            self.tabla_mu.limpiar()
            self.mapa.set_estado(d["mu_e"], d["mu_de"], d["regla"])
            self.curva_agregado.setData(_X_OUT, d["agregado"])
            self.marca_centro.setData([0], [0])
            self.cursor_sal.setValue(0)
            self.lbl_fuzzy.setText(
                "FUZZY DESHABILITADO: SP < 1 RPM → el firmware no calcula "
                "salida (ΔPWM = 0, PWM = 0)"
            )
            return

        self.barras_e.set_valores(d["mu_e"], d["e_dom"])
        self.barras_de.set_valores(d["mu_de"], d["de_dom"])
        self.barras_sal.set_valores(d["act"], d["sal_dom"])
        self.tabla_mu.set_datos(
            [d["mu_e"], d["mu_de"], d["act"]],
            [d["e_dom"], d["de_dom"], d["sal_dom"]],
        )

        self.cursor_e.setValue(d["e"])
        self.cursor_de.setValue(d["de"])
        self.cursor_sal.setValue(d["dpwm"])

        self.curva_agregado.setData(_X_OUT, d["agregado"])
        self.marca_centro.setData([d["dpwm"]], [0])
        self.mapa.set_estado(d["mu_e"], d["mu_de"], d["regla"])

        i, j = d["regla"]
        txt = (
            f"<b>e</b> = {d['e']:+.2f} RPM → <b><span style='color:{COLORES_ET[d['e_dom']]}'>"
            f"{ETIQUETAS[d['e_dom']]}</span></b> ({d['mu_e'][d['e_dom']]:.2f})"
            f" &nbsp;|&nbsp; <b>Δe</b> = {d['de']:+.2f} → "
            f"<b><span style='color:{COLORES_ET[d['de_dom']]}'>"
            f"{ETIQUETAS[d['de_dom']]}</span></b> ({d['mu_de'][d['de_dom']]:.2f})"
            f" &nbsp;|&nbsp; regla más fuerte: "
            f"<b>(e={ETIQUETAS[i]}, Δe={ETIQUETAS[j]}) → "
            f"<span style='color:{COLORES_ET[d['regla_out']]}'>"
            f"{ETIQUETAS[d['regla_out']]}</span></b> con grado {d['fuerza']:.2f}"
            f" &nbsp;|&nbsp; salida agregada: "
            f"<b><span style='color:{COLORES_ET[d['sal_dom']]}'>"
            f"{ETIQUETAS[d['sal_dom']]}</span></b> ({d['act'][d['sal_dom']]:.2f})"
            f" &nbsp;|&nbsp; centroide ΔPWM = <b>{d['dpwm']:+.2f}</b>"
            f" (KU = {self.sp_ku.value():.2f}) → PWM {self.pwm[-1]:.0f}"
        )

        if self.modo == "real":
            txt += (
                f"<br>modo real: Δe reconstruido de la telemetría (20 Hz) mientras "
                f"el ESP32 decide a 50 Hz · ΔPWM recibido = {self.dpwm[-1]:+.2f}"
            )

        self.lbl_fuzzy.setText(txt)

    # ------------------------------------------------------------------
    # Métricas
    # ------------------------------------------------------------------

    def mostrar_metricas(self, m):
        if m is None:
            for c in (self.m_mp, self.m_ts, self.m_ess, self.m_iae, self.m_itae):
                c.valor.setText("—")
            return

        self.m_mp.valor.setText(f"{m['mp']:.1f} %")
        self.m_ts.valor.setText(
            "no estab." if m["ts"] is None else f"{m['ts']:.2f} s"
        )
        self.m_ess.valor.setText(
            "—" if m["ess"] is None else f"{m['ess']:.2f} RPM"
        )
        self.m_iae.valor.setText(f"{m['iae']:.1f}")
        self.m_itae.valor.setText(f"{m['itae']:.1f}")

    def actualizar_metricas(self):
        if self.idx_esc is None or len(self.t) - self.idx_esc < 10:
            self.mostrar_metricas(None)
            return

        i = self.idx_esc
        sp = self.sp[i]

        if sp < 1.0:
            self.mostrar_metricas(None)
            return

        m = calcular_metricas(self.t[i:], self.rpm[i:], sp, self.y0)
        self.mostrar_metricas(m)

    # ------------------------------------------------------------------
    # Modo
    # ------------------------------------------------------------------

    def cambiar_modo(self, idx):
        nuevo = "sim" if idx == 0 else "real"

        if nuevo == "real" and serial is None:
            QMessageBox.warning(
                self, "pyserial", "Instala pyserial: pip install pyserial"
            )
            self.combo_modo.setCurrentIndex(0)
            return

        self.desconectar(silencioso=True)
        self.sim_activa = False
        self.btn_sim.setText("▶ Iniciar")

        self.modo = nuevo
        self.gb_sim.setVisible(nuevo == "sim")
        self.gb_real.setVisible(nuevo == "real")
        self.limpiar()

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------

    def aplicar_params_sim(self, *_):
        self.sim.K = self.sp_K.value()
        self.sim.tau = self.sp_tau.value()
        self.sim.zona_muerta = self.sp_zm.value()
        self.sim.carga = self.sp_carga.value()
        self.sim.cuantizar = self.chk_cuant.isChecked()

    def cambiar_velocidad(self, idx):
        self.velocidad = self.VELOCIDADES[idx]

    def toggle_sim(self):
        self.sim_activa = not self.sim_activa

        if self.sim_activa:
            self._ult_tick = time.perf_counter()
            self._acum = 0.0
            self.btn_sim.setText("⏸ Pausar")
        else:
            self.btn_sim.setText("▶ Iniciar")

    def reiniciar_sim(self):
        self.sim.reset()
        self.sim.ctrl.ku = self.sp_ku.value()
        self.limpiar()

    # ------------------------------------------------------------------
    # Real: serial
    # ------------------------------------------------------------------

    def detectar_puertos(self):
        r"""
        Lista solo puertos utilizables:

        * BTHENUM  -> puertos Bluetooth: abrirlos bloquea ~5 s y falla.
        * ROOT\PORTS -> dispositivo fantasma de Windows (Status=Error),
          aparece en la lista pero da FileNotFoundError al abrirlo.
        """
        self.combo_puerto.clear()
        self._puertos = {}

        if serial is None:
            return

        for p in serial.tools.list_ports.comports():
            hwid = (getattr(p, "hwid", "") or "").upper()

            if "BTHENUM" in hwid:
                continue

            if hwid.startswith("ROOT\\PORTS"):
                continue

            desc = (getattr(p, "description", "") or "").strip()
            texto = f"{p.device}  -  {desc}" if desc else p.device

            self.combo_puerto.addItem(texto, p.device)
            self._puertos[texto] = p.device

    def marcar_estado(self, texto, color=""):
        self.estado.setText(texto)
        self.estado.setStyleSheet(f"font-weight:bold; {color}")

    def conectar(self):
        puerto = self.combo_puerto.currentData()

        if not puerto and self.combo_puerto.currentText().strip():
            puerto = self.combo_puerto.currentText().split()[0]

        puerto = (puerto or "").strip()

        if not puerto or serial is None:
            return

        self.desconectar(silencioso=True)

        try:
            # timeout=0 -> no bloquea; ademas se baja DTR/RTS para que el
            # ESP32 NO se reinicie (clave en ESP32-S3/C3 con USB nativo).
            self.enlace = EnlaceESP32(puerto, 115200)
        except Exception as e:
            self.enlace = None
            self.marcar_estado("ERROR AL ABRIR", "color:#c62828;")
            self.detalle.setText(f"{type(e).__name__}: {e}")
            return

        self.esp32_listo = False
        self._t_ultimo = None
        self._lineas_malas = 0
        self.marcar_estado("ABIERTO - ESP32 reiniciando", "color:#ef6c00;")
        self.detalle.setText("esperando banner FUZZY_MOTOR_READY ...")
        self.limpiar()

    def desconectar(self, silencioso=False, error=False):
        if self.enlace:
            self.enlace.cerrar()
            self.enlace = None

        self.esp32_listo = False
        self.sp_deseado = 0.0

        if error:
            self.marcar_estado("ERROR DE ENLACE", "color:#c62828;")
        elif not silencioso:
            self.marcar_estado("DESCONECTADO")
            self.detalle.setText("")

    def _enviar(self, texto):
        if not self.enlace:
            return False

        ok = self.enlace.escribir(texto)

        if not ok:
            err = self.enlace.error
            self.marcar_estado("ERROR AL ENVIAR", "color:#c62828;")
            self.detalle.setText(f"{type(err).__name__}: {err}" if err else "")
            self.desconectar(error=True)

        return ok

    def enviar_ku(self):
        if self.modo == "real" and self.enlace:
            self.enlace.escribir(f"KU={self.sp_ku.value():.2f}")

    def handshake(self):
        """Se dispara al ver el banner: el ESP32 ya tiene Serial listo."""
        ya_estaba = self.esp32_listo
        self.esp32_listo = True

        if ya_estaba and time.perf_counter() - self._t_ping < 1.0:
            # banner duplicado: no repetir el envio
            return

        self.marcar_estado("CONECTADO", "color:#2e7d32;")
        self.enviar_ku()
        if self.sp_deseado > 0.0:
            self._enviar(f"SP={self.sp_deseado:.1f}")
        self._t_envio_sp = time.perf_counter()
        self._t_ping = self._t_envio_sp

    def _rebase_t(self, t_abs):
        """
        Devuelve el tiempo relativo. Si el ESP32 se reinicio, T vuelve a 0 y
        la grafica se partiria en negativo, asi que se desplaza la base para
        que la linea de tiempo siga siendo monotona.
        """
        if self.t0_real is None:
            self.t0_real = t_abs
            return 0.0

        t = t_abs - self.t0_real

        if self._t_ultimo is not None and t <= self._t_ultimo:
            self.t0_real = t_abs - (self._t_ultimo + 0.05)
            t = t_abs - self.t0_real

        return t

    def verificar_setpoint(self):
        """
        El ESP32 reenvia su SP en cada telemetria. Si lo que reporta no
        coincide con el pedido, el comando se perdio (normal cuando el
        ESP32 se reinicia por DTR): se reenvia solo.
        """
        if not self.esp32_listo or not self.sp:
            return

        if abs(self.sp[-1] - self.sp_deseado) > 0.5:
            ahora = time.perf_counter()
            if ahora - self._t_envio_sp > 0.8:
                self._t_envio_sp = ahora
                self._enviar(f"SP={self.sp_deseado:.1f}")

    def leer_serial(self):
        if not self.enlace:
            return False

        nuevos = False

        for linea in self.enlace.leer():
            if self.procesar_linea(linea):
                nuevos = True

        if self.enlace.error is not None:
            err = self.enlace.error

            # Un fallo puntual no corta el enlace: solo si se repite
            if self.enlace.fallos >= 5:
                self.marcar_estado("ERROR DE ENLACE", "color:#c62828;")
                self.detalle.setText(f"{type(err).__name__}: {err}")
                self.desconectar(error=True)
                return False

        self.verificar_setpoint()

        return nuevos

    def procesar_linea(self, linea):
        # Banner de arranque: el ESP32 ya ejecuta setup() y tiene Serial listo
        if EnlaceESP32.BANNER in linea:
            self.handshake()
            return False

        # Ruido de arranque del ESP32 (ROM/IDF): se cuenta y se descarta
        if "SP=" not in linea or "RPM=" not in linea:
            self._lineas_malas += 1
            return False

        datos = {}

        for parte in linea.split(","):
            k, _, v = parte.partition("=")
            try:
                datos[k.strip()] = float(v)
            except ValueError:
                datos = {}
                break

        if not datos:
            self._lineas_malas += 1
            return False

        try:
            if "T" in datos:
                t = self._rebase_t(datos["T"] / 1000.0)
            else:
                if self.t0_real is None:
                    self.t0_real = time.perf_counter()
                t = time.perf_counter() - self.t0_real

            self._t_ultimo = t

            self.registrar(
                t,
                datos["SP"],
                datos["RPM"],
                datos["PWM"],
                datos["ERR"],
                datos.get("DPWM", 0.0),
            )
            return True

        except Exception:
            self._lineas_malas += 1
            return False

    def watchdog_enlace(self):
        """Si el ESP32 deja de emitir telemetria, avisar en vez de quedar mudo."""
        if self.modo != "real" or not self.enlace:
            return

        if not self.esp32_listo:
            if self.enlace.sin_datos_desde() > 6.0:
                self.marcar_estado("SIN BANNER", "color:#c62828;")
                self.detalle.setText(
                    "el ESP32 no envie FUZZY_MOTOR_READY: "
                    "revisa cable, baudios y que el sketch este cargado"
                )
            return

        silencio = self.enlace.sin_datos_desde()

        if silencio > 1.0:
            self.marcar_estado("CONECTADO - SIN DATOS", "color:#ef6c00;")
            self.detalle.setText(f"sin telemetria hace {silencio:.1f} s")
        else:
            self.marcar_estado("CONECTADO", "color:#2e7d32;")
            self.detalle.setText(
                f"{len(self.t)} muestras | "
                f"{self.enlace.lineas_malas + self._lineas_malas} lineas descartadas"
            )

    # ------------------------------------------------------------------
    # Acciones comunes
    # ------------------------------------------------------------------

    def enviar_setpoint(self):
        v = float(self.slider.value())
        self.sp_deseado = v

        if self.modo == "sim":
            self.sim.sp = v
        else:
            self._t_envio_sp = time.perf_counter()

            if self.esp32_listo:
                self._enviar(f"SP={v:.1f}")
            else:
                # El ESP32 aun no esta listo: se mandara al ver el banner
                self.marcar_estado("SP pendiente", "color:#ef6c00;")
                self.detalle.setText("se enviara al recibir FUZZY_MOTOR_READY")

    def enviar_stop(self):
        if self.modo == "sim":
            self.sim.sp = 0.0
            self.sim.pwm = 0.0
        else:
            self.sp_deseado = 0.0
            self._enviar("STOP")

    def cambiar_ku(self, v):
        self.sim.ctrl.ku = v
        self.fz_view.ku = v

        if self.modo == "real":
            self._enviar(f"KU={v:.2f}")

    def guardar_csv(self):
        if not self.t:
            QMessageBox.information(self, "CSV", "No hay datos para guardar.")
            return

        sugerido = f"fuzzy_{self.modo}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        nombre, _ = QFileDialog.getSaveFileName(
            self, "Guardar datos", sugerido, "CSV (*.csv)"
        )

        if not nombre:
            return

        with open(nombre, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["t_s", "sp_rpm", "rpm", "pwm", "err_rpm", "de_rpm", "dpwm"]
                + [f"mu_e_{k}" for k in ETIQUETAS]
                + [f"mu_de_{k}" for k in ETIQUETAS]
                + [f"mu_out_{k}" for k in ETIQUETAS]
            )
            for i in range(len(self.t)):
                fila = [
                    f"{self.t[i]:.4f}",
                    f"{self.sp[i]:.2f}",
                    f"{self.rpm[i]:.2f}",
                    f"{self.pwm[i]:.1f}",
                    f"{self.err[i]:.3f}",
                    f"{self.de[i]:.3f}",
                    f"{self.dpwm[i]:.3f}",
                ]
                fila += [f"{v:.4f}" for v in self.mu_e[i]]
                fila += [f"{v:.4f}" for v in self.mu_de[i]]
                fila += [f"{v:.4f}" for v in self.mu_sal[i]]
                w.writerow(fila)

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------

    def tick(self):
        try:
            if self.modo == "sim":
                if not self.sim_activa:
                    return

                ahora = time.perf_counter()
                self._acum += (ahora - self._ult_tick) * self.velocidad
                self._ult_tick = ahora

                n = min(int(self._acum / TS), 400)
                self._acum -= n * TS

                for _ in range(n):
                    self.registrar(*self.sim.paso())

                if n:
                    self.refrescar()
            else:
                if self.leer_serial():
                    self.refrescar()

        except Exception as e:
            # Nunca dejar que un error puntual congele la interfaz
            self.marcar_estado("ERROR INTERNO", "color:#c62828;")
            self.detalle.setText(f"{type(e).__name__}: {e}")

    def closeEvent(self, event):
        self.desconectar(silencioso=True)
        super().closeEvent(event)


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":

    if "--figuras" in sys.argv:
        generar_figuras()
        sys.exit(0)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    ventana = HMI()
    ventana.show()

    sys.exit(app.exec())
