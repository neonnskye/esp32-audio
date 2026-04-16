#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUDP.h>
#include "esp_wifi.h"
#include "driver/adc.h"

// ---- User configuration ----
#define WIFI_SSID "Slt2657"
#define WIFI_PASSWORD "Amrith@123"
#define PC_IP "192.168.1.54"
#define UDP_PORT 12345
// ----------------------------

#define SAMPLES_PER_PKT 512

// Double buffer: ISR writes to one half, main loop sends the other
uint16_t buf[2][SAMPLES_PER_PKT];
volatile int writeBuf = 0;
volatile int writeIdx = 0;
volatile int readyBuf = -1;

portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
hw_timer_t *timer = NULL;

// ISR: fires at exactly 16 000 Hz, reads one ADC sample, swaps buffer when full.
// adc1_get_raw() is called outside the spinlock to keep the critical section minimal.
void IRAM_ATTR onTimer()
{
    uint16_t sample = (uint16_t)adc1_get_raw(ADC1_CHANNEL_7);
    portENTER_CRITICAL_ISR(&mux);
    buf[writeBuf][writeIdx++] = sample;
    if (writeIdx >= SAMPLES_PER_PKT)
    {
        writeIdx = 0;
        readyBuf = writeBuf;
        writeBuf ^= 1;
    }
    portEXIT_CRITICAL_ISR(&mux);
}

WiFiUDP udp;
uint32_t packetsSent = 0;
uint32_t packetsFailed = 0;

void setup()
{
    Serial.begin(115200);

    // Configure ADC via IDF API (GPIO 35 = ADC1 channel 7)
    // 12-bit resolution, 11 dB attenuation = full 0–3.3 V input range
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC1_CHANNEL_7, ADC_ATTEN_DB_12);

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("Connected, IP: ");
    Serial.println(WiFi.localIP());

    // Give lwIP time to populate ARP table and prepare UDP send buffers
    Serial.println("Waiting for network to stabilize...");
    delay(2000);

    // Disable WiFi modem sleep to reduce ADC interference from radio bursts
    esp_wifi_set_ps(WIFI_PS_NONE);

    // Timer 0: 80 MHz / prescaler 5 = 16 MHz base clock
    // Alarm at 1000 counts → 16 MHz / 1000 = exactly 16 000 Hz
    timer = timerBegin(0, 5, true);
    timerAttachInterrupt(timer, &onTimer, true);
    timerAlarmWrite(timer, 1000, true);
    timerAlarmEnable(timer);

    Serial.println("Streaming audio...");
    Serial.print("Sent: ");
    Serial.print(packetsSent);
    Serial.print(" | Failed: ");
    Serial.println(packetsFailed);
}

void loop()
{
    if (readyBuf < 0)
        return;

    int toSend;
    portENTER_CRITICAL(&mux);
    toSend = readyBuf;
    portEXIT_CRITICAL(&mux);

    if (udp.beginPacket(PC_IP, UDP_PORT) == 0)
    {
        // beginPacket failed — will retry next iteration
        packetsFailed++;
        return;
    }

    udp.write((uint8_t *)buf[toSend], SAMPLES_PER_PKT * sizeof(uint16_t));

    if (udp.endPacket() != 0)
    {
        // Send succeeded — mark buffer as consumed
        portENTER_CRITICAL(&mux);
        readyBuf = -1;
        portEXIT_CRITICAL(&mux);
        packetsSent++;
        yield();
    }
    else
    {
        // Send failed — packet not consumed, will retry next iteration
        packetsFailed++;
    }
}
