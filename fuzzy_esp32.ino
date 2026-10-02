// ======================================================
// CONTROL FUZZY DE VELOCIDAD - MOTOR DC CON ENCODER (ESP32)
// Compatible con fuzzy_hmi.py (modo Real)
//
// Controlador: Fuzzy PI incremental (Mamdani)
//   Entradas  : error E y variación de error dE
//   Salida    : incremento de PWM (dPWM), que se integra
//   Inferencia: AND = min, implicación = min, agregación = max
//   Defuzz.   : centroide (discretizado, paso 1)
//
// CABLEADO (L298N con el jumper de ENA puesto)
//   GPIO 4  <- Encoder canal A (cable amarillo)
//   GPIO 5  -> IN1 del L298N   (recibe el PWM)
//   GPIO 6  -> IN2 del L298N   (queda siempre en LOW)
//   Si el motor gira al revés: poner INVERTIR_GIRO = true
//
// Comandos por serial (terminados en \n):
//   SP=<rpm>   setpoint
//   KU=<x>     ganancia de salida
//   STOP       detiene el motor
//   RESET      detiene y reinicia estados
//
// Protocolo con la HMI:
//   - Envía "FUZZY_MOTOR_READY" al arrancar y cada 2 s (la HMI espera
//     esa línea para mandar SP y KU).
//   - Telemetría a 20 Hz:
//     T=<ms>,SP=..,RPM=..,PWM=..,ERR=..,DPWM=..
// ======================================================

#include <Arduino.h>

// ======================================================
// PINES
// ======================================================

const int PIN_ENC = 4;   // Encoder canal A
const int PIN_IN1 = 5;   // L298N IN1
const int PIN_IN2 = 6;   // L298N IN2

// false: PWM por IN1, IN2 en LOW
// true : PWM por IN2, IN1 en LOW (invierte el sentido de giro)
const bool INVERTIR_GIRO = false;

int pinPWM = PIN_IN1;
int pinLow = PIN_IN2;

// ======================================================
// MOTOR / ENCODER
// ======================================================

// VALIDAR: girar el eje 1 vuelta a mano y contar pulsos
const float PULSOS_POR_VUELTA = 858.0;

const float RPM_MAX = 100.0;

// Antirrebote: ignora pulsos más cercanos que esto
const uint32_t MIN_INTERVALO_PULSO_US = 100;

// ======================================================
// TIEMPOS
// ======================================================

const uint32_t DT_CONTROL_US = 20000;     // 50 Hz
const uint32_t DT_SERIAL_US  = 50000;     // 20 Hz
const uint32_t DT_BANNER_MS  = 2000;      // latido del banner

// Ventana deslizante para medir RPM (5 muestras = 100 ms)
const int N_VENTANA = 5;

// ======================================================
// UNIVERSOS DEL FUZZY (deben coincidir con fuzzy_hmi.py)
// ======================================================

const float E_MAX  = 100.0;   // error   [-100, 100] RPM
const float DE_MAX = 10.0;    // dError  [-10, 10]  RPM/muestra
float KU = 0.5;               // ganancia de salida

// ======================================================
// ESTADO
// ======================================================

float setpoint = 0.0;
float rpm = 0.0;
float error = 0.0;
float errorAnterior = 0.0;
float deltaError = 0.0;
float pwm = 0.0;
float deltaPWM = 0.0;

// ======================================================
// ENCODER (ISR con antirrebote)
// ======================================================

volatile uint32_t pulsos = 0;
volatile uint32_t ultimoPulsoUs = 0;
portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;

void IRAM_ATTR contarPulso()
{
    uint32_t t = micros();

    portENTER_CRITICAL_ISR(&mux);

    if ((uint32_t)(t - ultimoPulsoUs) >= MIN_INTERVALO_PULSO_US)
    {
        pulsos++;
        ultimoPulsoUs = t;
    }

    portEXIT_CRITICAL_ISR(&mux);
}

// ======================================================
// FUNCIONES DE MEMBRESÍA
// ======================================================

float leftShoulder(float x, float a, float b)
{
    if (x <= a) return 1.0;
    if (x >= b) return 0.0;
    return (b - x) / (b - a);
}

float rightShoulder(float x, float a, float b)
{
    if (x <= a) return 0.0;
    if (x >= b) return 1.0;
    return (x - a) / (b - a);
}

float trimf(float x, float a, float b, float c)
{
    if (x <= a || x >= c)
        return (x == b) ? 1.0 : 0.0;

    if (x == b) return 1.0;
    if (x < b)  return (x - a) / (b - a);

    return (c - x) / (c - b);
}

// m[0]=NB  m[1]=NS  m[2]=ZE  m[3]=PS  m[4]=PB
void memberships(float x, float limit, float m[5])
{
    x = constrain(x, -limit, limit);

    m[0] = leftShoulder(x, -limit, -0.35 * limit);
    m[1] = trimf(x, -0.75 * limit, -0.30 * limit, 0);
    m[2] = trimf(x, -0.20 * limit, 0, 0.20 * limit);
    m[3] = trimf(x, 0, 0.30 * limit, 0.75 * limit);
    m[4] = rightShoulder(x, 0.35 * limit, limit);
}

// Membresías de salida (dPWM en [-20, 20])
float outputMF(int label, float x)
{
    if (label == 0) return leftShoulder(x, -20, -7);   // NB
    if (label == 1) return trimf(x, -15, -7, 0);       // NS
    if (label == 2) return trimf(x, -4, 0, 4);         // ZE
    if (label == 3) return trimf(x, 0, 7, 15);         // PS
    return rightShoulder(x, 7, 20);                    // PB
}

// ======================================================
// BASE DE REGLAS (filas = E, columnas = dE)
// 0=NB 1=NS 2=ZE 3=PS 4=PB
// ======================================================

const int rules[5][5] =
{
    // dE: NB NS ZE PS PB
    {0, 0, 0, 1, 2},   // E = NB
    {0, 1, 1, 2, 3},   // E = NS
    {0, 1, 2, 3, 4},   // E = ZE
    {1, 2, 3, 3, 4},   // E = PS
    {2, 3, 4, 4, 4}    // E = PB
};

// ======================================================
// CONTROLADOR FUZZY (Mamdani, centroide)
// ======================================================

float fuzzyController(float e, float de)
{
    float eM[5];
    float dM[5];

    memberships(e,  E_MAX,  eM);
    memberships(de, DE_MAX, dM);

    // Activación máxima por etiqueta de salida
    float act[5] = {0, 0, 0, 0, 0};

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            int s = rules[i][j];
            float a = min(eM[i], dM[j]);
            if (a > act[s]) act[s] = a;
        }
    }

    float numerador = 0.0;
    float denominador = 0.0;

    for (int xi = -20; xi <= 20; xi++)
    {
        float x = (float)xi;
        float agregado = 0.0;

        for (int k = 0; k < 5; k++)
        {
            float r = min(act[k], outputMF(k, x));
            if (r > agregado) agregado = r;
        }

        numerador   += x * agregado;
        denominador += agregado;
    }

    if (denominador <= 0.000001)
        return 0.0;

    return KU * (numerador / denominador);
}

// ======================================================
// MEDICIÓN DE RPM (ventana deslizante, dt real)
// ======================================================

uint32_t ventPulsos[N_VENTANA];
float    ventDt[N_VENTANA];
int      ventIdx = 0;
int      ventN = 0;

float medirRPM(uint32_t pulsosMuestra, float dt)
{
    ventPulsos[ventIdx] = pulsosMuestra;
    ventDt[ventIdx] = dt;

    ventIdx = (ventIdx + 1) % N_VENTANA;

    if (ventN < N_VENTANA)
        ventN++;

    uint32_t sumaPulsos = 0;
    float sumaDt = 0.0;

    for (int i = 0; i < ventN; i++)
    {
        sumaPulsos += ventPulsos[i];
        sumaDt += ventDt[i];
    }

    if (sumaDt <= 0.0)
        return 0.0;

    return ((float)sumaPulsos / PULSOS_POR_VUELTA) * (60.0 / sumaDt);
}

// ======================================================
// COMANDOS SERIAL (no bloqueante)
// ======================================================

void detenerMotor()
{
    setpoint = 0.0;
    pwm = 0.0;
    deltaPWM = 0.0;
    analogWrite(pinPWM, 0);
}

void procesarComando(const char *c)
{
    if (strncmp(c, "SP=", 3) == 0)
    {
        float v = (float)atof(c + 3);
        setpoint = constrain(v, 0.0, RPM_MAX);
    }
    else if (strncmp(c, "KU=", 3) == 0)
    {
        float v = (float)atof(c + 3);
        KU = constrain(v, 0.05, 3.0);
    }
    else if (strcmp(c, "STOP") == 0)
    {
        detenerMotor();
    }
    else if (strcmp(c, "RESET") == 0)
    {
        detenerMotor();
        error = 0.0;
        errorAnterior = 0.0;
        deltaError = 0.0;
        ventN = 0;
        ventIdx = 0;
        rpm = 0.0;
    }
}

void leerSerial()
{
    static char buf[32];
    static uint8_t n = 0;

    while (Serial.available())
    {
        char c = (char)Serial.read();

        if (c == '\n')
        {
            buf[n] = 0;
            procesarComando(buf);
            n = 0;
        }
        else if (c != '\r' && n < sizeof(buf) - 1)
        {
            buf[n++] = c;
        }
    }
}

// ======================================================
// SETUP
// ======================================================

uint32_t tControlUs = 0;
uint32_t tSerialUs = 0;
uint32_t tBannerMs = 0;

void setup()
{
    Serial.begin(115200);

    // Con USB nativo (S3/C3) evita que Serial.print bloquee el lazo de
    // control cuando la PC no está leyendo el puerto.
#if defined(ARDUINO_USB_CDC_ON_BOOT) && (ARDUINO_USB_CDC_ON_BOOT == 1)
    Serial.setTxTimeoutMs(0);
#endif

    if (INVERTIR_GIRO)
    {
        pinPWM = PIN_IN2;
        pinLow = PIN_IN1;
    }

    pinMode(pinPWM, OUTPUT);
    pinMode(pinLow, OUTPUT);
    pinMode(PIN_ENC, INPUT_PULLUP);

    digitalWrite(pinLow, LOW);
    analogWrite(pinPWM, 0);

    attachInterrupt(
        digitalPinToInterrupt(PIN_ENC),
        contarPulso,
        RISING
    );

    Serial.println("FUZZY_MOTOR_READY");

    tControlUs = micros();
    tSerialUs = tControlUs;
    tBannerMs = millis();
}

// ======================================================
// LOOP
// ======================================================

void loop()
{
    leerSerial();

    uint32_t ahora = micros();

    // ==================================================
    // LAZO DE CONTROL (50 Hz, dt real)
    // ==================================================

    if ((uint32_t)(ahora - tControlUs) >= DT_CONTROL_US)
    {
        float dt = (float)(ahora - tControlUs) / 1000000.0;
        tControlUs = ahora;

        portENTER_CRITICAL(&mux);
        uint32_t pulsosMuestra = pulsos;
        pulsos = 0;
        portEXIT_CRITICAL(&mux);

        rpm = medirRPM(pulsosMuestra, dt);
        rpm = constrain(rpm, 0.0, RPM_MAX * 1.5);

        error = setpoint - rpm;
        deltaError = error - errorAnterior;

        if (setpoint < 1.0)
        {
            deltaPWM = 0.0;
            pwm = 0.0;
        }
        else
        {
            deltaPWM = fuzzyController(error, deltaError);
            pwm = constrain(pwm + deltaPWM, 0.0, 255.0);
        }

        analogWrite(pinPWM, (int)pwm);

        errorAnterior = error;
    }

    // ==================================================
    // TELEMETRÍA (20 Hz)
    // ==================================================

    if ((uint32_t)(ahora - tSerialUs) >= DT_SERIAL_US)
    {
        tSerialUs = ahora;

        char linea[110];

        snprintf(
            linea,
            sizeof(linea),
            "T=%lu,SP=%.2f,RPM=%.2f,PWM=%.2f,ERR=%.2f,DPWM=%.2f",
            (unsigned long)millis(),
            setpoint,
            rpm,
            pwm,
            error,
            deltaPWM
        );

        Serial.println(linea);
    }

    // ==================================================
    // LATIDO DEL BANNER (la HMI lo usa para el handshake)
    // ==================================================

    uint32_t ms = millis();

    if ((uint32_t)(ms - tBannerMs) >= DT_BANNER_MS)
    {
        tBannerMs = ms;
        Serial.println("FUZZY_MOTOR_READY");
    }
}
