#include <Arduino.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <ESPmDNS.h>
#include <WiFiUDP.h>
#include "esp_wifi.h"
#include "driver/adc.h"
#include <driver/i2s.h>
#include "jbl_begin.h"
#include "jbl_latency.h"

#include <PubSubClient.h>

#define EIDSP_QUANTIZE_FILTERBANK 0
#include <Elio_Wake_v3.1_inferencing.h>

// ---- User configuration ----
#define WIFI_SSID_1    "Amrith’s iPhone"
#define WIFI_PASS_1    "brat summer"
#define WIFI_SSID_2    "Slt2657"
#define WIFI_PASS_2    "Amrith@123"
#define PC_MDNS_HOST   "raspberrypi"
#define ESP32_MDNS_HOST "esp32-audio"
#define MQTT_BROKER    "127.0.0.1"   // broker runs locally on the Pi
#define MQTT_PORT           1883
#define MQTT_ID             "elio-esp32"
#define TOPIC_WAKE          "elio/wake"
#define TOPIC_CTRL          "elio/ctrl"
#define UDP_PORT 12345
#define AUDIO_RX_PORT 12347      // PC sends TTS audio back to this port
#define PLAYBACK_VOLUME_PCT 95   // volume scale applied to each sample (out of 100)
#define CHIME_VOLUME_PCT 95      // volume scale applied to chime samples (out of 100)
// ----------------------------

// GPIO 2 is the standard built-in LED on most ESP32 devboards.
// The generic ESP32_DEV variant does not define LED_BUILTIN, so we define it here.
#ifndef LED_BUILTIN
#define LED_BUILTIN 2
#endif

#define SAMPLES_PER_PKT 512

// Double buffer: ISR writes to one half, main loop sends the other
uint16_t buf[2][SAMPLES_PER_PKT];
volatile int writeBuf = 0;
volatile int writeIdx = 0;
volatile int readyBuf = -1;

portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
hw_timer_t *timer = NULL;

volatile bool isSpeaking = false;
volatile bool isListening = false; // true from wake word until VAD end (0x03)
volatile bool chimeLooping = false;
static TaskHandle_t chimeTaskHandle = NULL;

// EI inference double-buffer (fed from the timer ISR, consumed by inference task)
typedef struct
{
    int16_t *buffers[2];
    volatile uint8_t buf_select;
    volatile uint8_t buf_ready;
    volatile uint32_t buf_count;
    uint32_t n_samples;
} ei_inference_t;

static ei_inference_t ei_inf;
static TaskHandle_t inferenceTaskHandle = NULL;

// ISR: fires at exactly 16 000 Hz, reads one ADC sample, swaps buffer when full.
// adc1_get_raw() is called outside the spinlock to keep the critical section minimal.
void IRAM_ATTR onTimer()
{
    uint16_t sample = (uint16_t)adc1_get_raw(ADC1_CHANNEL_7);

    // Do arithmetic OUTSIDE the lock
    // (cheap optimization, avoids extra work while holding spinlock)
    int16_t s16 = (int16_t)(sample - 2048) * 16;

    portENTER_CRITICAL_ISR(&mux);

    // UDP buffer
    buf[writeBuf][writeIdx++] = sample;
    if (writeIdx >= SAMPLES_PER_PKT)
    {
        writeIdx = 0;
        readyBuf = writeBuf;
        writeBuf ^= 1;
    }

    // EI inference buffer
    ei_inf.buffers[ei_inf.buf_select][ei_inf.buf_count++] = s16;

    if (ei_inf.buf_count >= ei_inf.n_samples)
    {
        ei_inf.buf_select ^= 1;
        ei_inf.buf_count = 0;

        // Notify inference task instead of setting a flag
        BaseType_t xHigherPriorityTaskWoken = pdFALSE;
        vTaskNotifyGiveFromISR(inferenceTaskHandle, &xHigherPriorityTaskWoken);

        if (xHigherPriorityTaskWoken)
        {
            portYIELD_FROM_ISR();
        }
    }

    portEXIT_CRITICAL_ISR(&mux);
}

static int ei_get_data(size_t offset, size_t length, float *out_ptr)
{
    uint8_t done_buf = ei_inf.buf_select ^ 1;
    for (size_t i = 0; i < length; i++)
    {
        out_ptr[i] = (float)ei_inf.buffers[done_buf][offset + i] / 2048.0f;
    }
    return 0;
}

static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

WiFiUDP udp;

WiFiClient  wifiClient;
PubSubClient mqttClient(wifiClient);
WiFiMulti wifiMulti;
IPAddress pcIP;

void playChime(const int16_t *samples,
               uint32_t length_bytes,
               volatile bool *keepPlaying = NULL)
{
    isSpeaking = true;

    const uint32_t CHUNK_SAMPLES = SAMPLES_PER_PKT * 2; // stereo: 2 int16 per frame
    const uint32_t CHUNK_BYTES = CHUNK_SAMPLES * sizeof(int16_t);
    static int16_t chimeBuf[SAMPLES_PER_PKT * 2];

    uint32_t offset = 0;
    while (offset < length_bytes)
    {
        if (keepPlaying && !(*keepPlaying))
            break;

        uint32_t toRead = min(CHUNK_BYTES, length_bytes - offset);
        uint32_t sampleCount = toRead / sizeof(int16_t);

        const int16_t *src = samples + (offset / sizeof(int16_t));
        for (uint32_t i = 0; i < sampleCount; i++)
        {
            chimeBuf[i] = (int16_t)((int32_t)src[i] * CHIME_VOLUME_PCT / 100);
        }

        size_t bytesWritten;
        i2s_write(I2S_NUM_0, chimeBuf, toRead, &bytesWritten, portMAX_DELAY);
        offset += toRead;
    }

    isSpeaking = false;
}

void inferenceTask(void *arg)
{
    static bool debug_nn = false;

    while (true)
    {
        // Block indefinitely until the ISR sends a notification
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        signal_t signal;
        signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
        signal.get_data = &ei_get_data;

        ei_impulse_result_t result = {0};
        EI_IMPULSE_ERROR err = run_classifier_continuous(&signal, &result, debug_nn);
        if (err != EI_IMPULSE_OK)
        {
            Serial.printf("EI classifier error: %d\n", err);
            continue;
        }

        if (++print_results >= EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)
        {
            for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
            {
                Serial.printf("  %s: %.2f\n",
                              result.classification[i].label,
                              result.classification[i].value);

                if (strcmp(result.classification[i].label, "elio") == 0 &&
                    result.classification[i].value > 0.6f)
                {
                    Serial.println(">>> WAKE WORD DETECTED <<<");

                    // Notify Python server that wake word was detected
                    // Suppress the trigger while ESP32 is playing its own audio
                    // (prevents acoustic feedback through the microphone)
                    if (!isSpeaking)
                    {
                        // Turn on LED immediately; it will stay on until the
                        // PC signals "stop" via MQTT on elio/ctrl, at which point
                        // mqttCallback turns it off.
                        isListening = true;
                        digitalWrite(LED_BUILTIN, HIGH);

                        // Notify PC first so it starts its bleed-skip window
                        // immediately. The chime plays after — its duration is
                        // covered by BLEED_SKIP_PACKETS on the receiver side,
                        // so no speech is lost.
                        mqttClient.publish(TOPIC_WAKE, "1");

                        playChime(jbl_begin, jbl_begin_length);
                    }
                }
            }
            print_results = 0;
        }
    }
}

void audioPlaybackTask(void *arg)
{
    WiFiUDP audioRxUdp;
    audioRxUdp.begin(AUDIO_RX_PORT);

    // Jitter buffer: hold this many packets before we start draining
    const int JITTER_BUFFER_PKTS = 3; // ~96ms at 16kHz/512 samples
    static int16_t jitterBuf[JITTER_BUFFER_PKTS][SAMPLES_PER_PKT];
    static int16_t stereoBuf[SAMPLES_PER_PKT * 2];
    int buffered = 0;
    bool playing = false;
    uint32_t lastPacketMs = 0;

    while (true)
    {
        int packetSize = audioRxUdp.parsePacket();

        if (packetSize > 0)
        {
            int16_t rxBuf[SAMPLES_PER_PKT];
            int bytesRead = audioRxUdp.read((uint8_t *)rxBuf, sizeof(rxBuf));
            int samplesRead = bytesRead / sizeof(int16_t);

            if (chimeLooping)
            {
                chimeLooping = false;

                while (chimeTaskHandle != NULL)
                {
                    vTaskDelay(1);
                }

                i2s_zero_dma_buffer(I2S_NUM_0);
            }
            isSpeaking = true;
            lastPacketMs = millis();

            // Scale volume
            for (int i = 0; i < samplesRead; i++)
                rxBuf[i] = (int16_t)((int32_t)rxBuf[i] * PLAYBACK_VOLUME_PCT / 100);

            if (!playing)
            {
                // Pre-fill jitter buffer before starting playback
                if (buffered < JITTER_BUFFER_PKTS)
                {
                    memcpy(jitterBuf[buffered++], rxBuf, samplesRead * sizeof(int16_t));
                    continue;
                }
                // Drain the pre-fill buffer first
                for (int p = 0; p < JITTER_BUFFER_PKTS; p++)
                {
                    for (int i = 0; i < SAMPLES_PER_PKT; i++)
                    {
                        stereoBuf[i * 2] = jitterBuf[p][i];
                        stereoBuf[i * 2 + 1] = jitterBuf[p][i];
                    }
                    size_t bw;
                    i2s_write(I2S_NUM_0, stereoBuf, SAMPLES_PER_PKT * 4, &bw, portMAX_DELAY);
                }
                buffered = 0;
                playing = true;
            }

            // Normal path: write directly to I2S
            for (int i = 0; i < samplesRead; i++)
            {
                stereoBuf[i * 2] = rxBuf[i];
                stereoBuf[i * 2 + 1] = rxBuf[i];
            }
            size_t bytesWritten;
            i2s_write(I2S_NUM_0, stereoBuf, samplesRead * 4, &bytesWritten, portMAX_DELAY);
        }
        else
        {
            if (isSpeaking && (millis() - lastPacketMs > 200))
            {
                isSpeaking = false;
                playing = false; // reset for next utterance
                buffered = 0;
            }
            delay(1);
        }
    }
}

uint32_t packetsSent = 0;
uint32_t packetsFailed = 0;

void chimeLoopTask(void *arg)
{
    while (chimeLooping)
    {
        playChime(jbl_latency, jbl_latency_length, &chimeLooping);
    }
    chimeTaskHandle = NULL;
    vTaskDelete(NULL);
}

void mqttCallback(char* topic, byte* payload, unsigned int length)
{
    // Build a null-terminated string from the payload bytes
    char msg[32] = {0};
    unsigned int copy = (length < sizeof(msg) - 1) ? length : sizeof(msg) - 1;
    memcpy(msg, payload, copy);

    if (strcmp(topic, TOPIC_CTRL) != 0)
        return; // ignore any topic we didn't subscribe to

    if (strcmp(msg, "processing") == 0 && !isSpeaking)
    {
        // Same logic as the old byte == 0x02 branch
        chimeLooping = false;
        while (chimeTaskHandle != NULL)
            vTaskDelay(1);

        chimeLooping = true;
        xTaskCreatePinnedToCore(
            chimeLoopTask,
            "ChimeLoop",
            1024 * 4,
            NULL,
            1,
            &chimeTaskHandle,
            1);
    }
    else if (strcmp(msg, "stop") == 0)
    {
        // Same logic as the old byte == 0x03 branch
        chimeLooping = false;
        while (chimeTaskHandle != NULL)
            vTaskDelay(1);

        i2s_zero_dma_buffer(I2S_NUM_0);

        if (isListening)
        {
            isListening = false;
            digitalWrite(LED_BUILTIN, LOW);
        }
    }
}

void mqttReconnect()
{
    // Single attempt per call — loop() drives retries every iteration.
    // This keeps the loop() responsive (mDNS retries, UDP sends) even when
    // the broker is temporarily unreachable.
    if (mqttClient.connected())
        return;
    Serial.print("Connecting to MQTT broker...");
    if (mqttClient.connect(MQTT_ID))
    {
        Serial.println(" connected.");
        mqttClient.subscribe(TOPIC_CTRL);
    }
    else
    {
        Serial.printf(" failed (rc=%d). Will retry next loop.\n", mqttClient.state());
        delay(500); // brief back-off; non-blocking relative to loop() cadence
    }
}

void setup()
{
    Serial.begin(115200);

    // Configure ADC via IDF API (GPIO 35 = ADC1 channel 7)
    // 12-bit resolution, 11 dB attenuation = full 0–3.3 V input range
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC1_CHANNEL_7, ADC_ATTEN_DB_12);

    wifiMulti.addAP(WIFI_SSID_1, WIFI_PASS_1);
    wifiMulti.addAP(WIFI_SSID_2, WIFI_PASS_2);
    Serial.print("Connecting to WiFi");
    while (wifiMulti.run() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.printf("Connected to %s, IP: %s\n",
        WiFi.SSID().c_str(), WiFi.localIP().toString().c_str());

    // Start mDNS — advertise this device as esp32-audio.local
    if (!MDNS.begin(ESP32_MDNS_HOST))
        Serial.println("mDNS init failed");
    else
        Serial.printf("mDNS started: %s.local\n", ESP32_MDNS_HOST);

    // Resolve PC/Pi IP via mDNS.
    // We do a best-effort attempt here but do NOT halt if it fails —
    // resolvePcIP() will keep retrying from loop() until it succeeds.
    Serial.printf("Resolving %s.local via mDNS...\n", PC_MDNS_HOST);
    for (int attempt = 0; attempt < 15; attempt++)
    {
        pcIP = MDNS.queryHost(PC_MDNS_HOST);
        if (pcIP.toString() != "0.0.0.0") break;
        Serial.print(".");
        delay(1000);
    }
    if (pcIP.toString() == "0.0.0.0")
    {
        Serial.println("\nmDNS resolution failed at boot — will keep retrying in loop().");
        Serial.println("Start receiver.py on the Pi; the ESP32 will connect automatically.");
    }
    else
    {
        Serial.printf("\nResolved %s.local -> %s\n", PC_MDNS_HOST, pcIP.toString().c_str());
    }

    // Give lwIP time to populate ARP table and prepare UDP send buffers
    Serial.println("Waiting for network to stabilize...");
    delay(2000);

    // Disable WiFi modem sleep to reduce ADC interference from radio bursts
    esp_wifi_set_ps(WIFI_PS_NONE);

    // MQTT setup — only connect if we already have the broker IP.
    // If pcIP is still 0.0.0.0 (Pi not running), loop() will resolve it
    // and call mqttReconnect() automatically once the IP is known.
    mqttClient.setServer(pcIP, MQTT_PORT);
    mqttClient.setCallback(mqttCallback);
    if (pcIP.toString() != "0.0.0.0")
        mqttReconnect();

    ei_inf.n_samples = EI_CLASSIFIER_SLICE_SIZE;
    ei_inf.buffers[0] = (int16_t *)malloc(EI_CLASSIFIER_SLICE_SIZE * sizeof(int16_t));
    ei_inf.buffers[1] = (int16_t *)malloc(EI_CLASSIFIER_SLICE_SIZE * sizeof(int16_t));
    ei_inf.buf_select = 0;
    ei_inf.buf_count = 0;
    ei_inf.buf_ready = 0;

    run_classifier_init();

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    // Initialize I2S for audio playback (MAX98357 amp)
    // 16-bit 16 kHz mono, duplicated to both stereo channels
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = SAMPLES_PER_PKT,
        .use_apll = false,
        .tx_desc_auto_clear = true,
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = 26,
        .ws_io_num = 25,
        .data_out_num = 22,
        .data_in_num = I2S_PIN_NO_CHANGE,
    };

    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);
    i2s_zero_dma_buffer(I2S_NUM_0);

    // Start inference task on core 0 (WiFi/UDP runs on core 1 by default)
    xTaskCreatePinnedToCore(inferenceTask, "EI_Infer", 1024 * 48, NULL, 1, &inferenceTaskHandle, 0);

    // Start audio playback task on core 1 (same as WiFi — spends most time blocked on UDP recv)
    xTaskCreatePinnedToCore(audioPlaybackTask, "AudioRX", 1024 * 8, NULL, 3, NULL, 1);

    // v3 timer API: timerBegin takes the desired frequency directly (Hz).
    // timerAlarm replaces timerAlarmWrite + timerAlarmEnable:
    //   pass period = 1 tick at 16 000 Hz → fires at exactly 16 000 Hz.
    timer = timerBegin(16000);             // 16 000 Hz timer clock
    timerAttachInterrupt(timer, &onTimer); // no 'edge' argument in v3
    timerAlarm(timer, 1, true, 0);         // fire every 1 tick = 16 000 Hz, auto-reload, unlimited

    Serial.println("Streaming audio...");
    Serial.print("Sent: ");
    Serial.print(packetsSent);
    Serial.print(" | Failed: ");
    Serial.println(packetsFailed);
}

// Attempt a single mDNS resolution of the PC hostname.
// Returns true if pcIP was (re-)resolved successfully.
// Called at boot and from loop() whenever pcIP is still 0.0.0.0.
bool resolvePcIP()
{
    IPAddress resolved = MDNS.queryHost(PC_MDNS_HOST, 1000); // 1-second timeout
    if (resolved.toString() == "0.0.0.0")
        return false;
    if (resolved != pcIP)
    {
        Serial.printf("[mDNS] Resolved %s.local -> %s\n", PC_MDNS_HOST, resolved.toString().c_str());
        pcIP = resolved;
        // Update MQTT broker address and force a reconnect so it uses the new IP
        mqttClient.setServer(pcIP, MQTT_PORT);
        mqttClient.disconnect();
    }
    return true;
}

void loop()
{
    // If pcIP was not resolved at boot (Pi wasn't running yet), keep retrying
    // every 3 seconds until it succeeds. Everything else is gated on pcIP being valid.
    if (pcIP.toString() == "0.0.0.0")
    {
        static uint32_t lastResolveAttemptMs = 0;
        if (millis() - lastResolveAttemptMs > 3000)
        {
            lastResolveAttemptMs = millis();
            Serial.printf("[mDNS] Retrying %s.local resolution...\n", PC_MDNS_HOST);
            resolvePcIP();
        }
        delay(10);
        return; // nothing else can work without the PC IP
    }

    if (!mqttClient.connected())
        mqttReconnect();
    mqttClient.loop();

    if (readyBuf < 0)
    {
        delay(1);
        return;
    }

    // Atomically claim and clear readyBuf before doing any UDP work
    int toSend;
    portENTER_CRITICAL(&mux);
    toSend = readyBuf;
    readyBuf = -1; // clear immediately so ISR can reuse the slot
    portEXIT_CRITICAL(&mux);

    if (toSend < 0)
        return; // another core beat us here (defensive)

    if (udp.beginPacket(pcIP, UDP_PORT) == 0)
    {
        packetsFailed++;
        delay(5);
        return;
    }

    udp.write((uint8_t *)buf[toSend], SAMPLES_PER_PKT * sizeof(uint16_t));

    if (udp.endPacket() != 0)
    {
        packetsSent++;
    }
    else
    {
        packetsFailed++;
        delay(5);
    }

    static uint32_t lastPrint = 0;
    if (millis() - lastPrint > 5000)
    {
        Serial.print("Sent: ");
        Serial.print(packetsSent);
        Serial.print(" | Failed: ");
        Serial.println(packetsFailed);
        lastPrint = millis();
    }
}