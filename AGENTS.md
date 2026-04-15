# AGENTS.md

## Build & Flash

```bash
pio run               # build
pio run --target upload  # build + flash
pio device monitor    # serial monitor (115200 baud)
pio run --target clean # clean build
```

## Project Structure

- **Firmware:** `src/main.cpp` — ESP32 Arduino app; hardcoded WiFi SSID/password/PC IP at lines 8–10
- **Receiver:** `receiver.py` — Python UDP receiver; run with `uv run receiver.py`
- **Python deps:** managed via `uv sync`; deps are `numpy`, `sounddevice`

## Key Architecture

- **Timer ISR (`onTimer`)** fires at exactly 16 000 Hz (timer 0, prescaler 5, alarm 1000 from 80 MHz clock)
- **Double buffer** (512 × `uint16_t` per half): ISR writes to `buf[writeBuf]`, `loop()` sends `buf[readyBuf]`
- **ADC:** IDF API via `driver/adc.h` — `adc1_get_raw(ADC1_CHANNEL_7)` on GPIO 35; NOT Arduino `analogRead`
- **WiFi power save** disabled (`esp_wifi_set_ps(WIFI_PS_NONE)`) to prevent radio bursts corrupting ADC
- **UDP packets:** 1024 bytes = 512 × little-endian `uint16_t`; receiver subtracts 2048 DC offset and normalizes to `[-1, 1]`

## Python Receiver

```bash
uv sync
uv run receiver.py
```

Listens on `0.0.0.0:12345`. Waits for `PREBUFFER_PKTS=3` packets before playback starts.

## Configuration

Firmware WiFi/IP config is in `src/main.cpp` lines 8–10. Update before deploying to a new network.

Python receiver constants (must match firmware): `UDP_PORT=12345`, `SAMPLE_RATE=16000`, `SAMPLES_PER_PKT=512`.
