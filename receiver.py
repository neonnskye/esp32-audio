import collections
import queue
import socket
import sys
import threading
import time
from datetime import datetime
from enum import Enum, auto

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


def ts() -> str:
    """Return a human-readable timestamp string."""
    return datetime.now().strftime("[%H:%M:%S.%f")[:-3] + "]"


class ListenState(Enum):
    IDLE = auto()  # waiting for wake word signal
    SKIP_WAKEWORD_BLEED = (
        auto()
    )  # discarding audio bleed from the wake word utterance itself
    CAPTURING = auto()  # actively recording the user's command
    TRANSCRIBING = auto()  # Whisper is processing, block new captures


# ---- Configuration ----
UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 12345
SAMPLE_RATE = 16000
SAMPLES_PER_PKT = 512
PREBUFFER_PKTS = 3  # Packets to queue before starting playback (~96ms)
MAX_QUEUE_LEN = 10  # Drop oldest if queue grows beyond this (~320ms)
NOISE_GATE = 0  # RMS threshold below which a packet is muted (0 = off)
VAD_SILENCE_THRESHOLD = 0.03  # RMS threshold below which a packet is considered silence
# -----------------------

# Whisper config
WHISPER_MODEL = "turbo"  # tiny / base / small — tradeoff speed vs accuracy
WHISPER_DEVICE = "cuda"  # or "cuda" if you have a GPU
WHISPER_COMPUTE = "float16"  # int8 = fastest on CPU

# VAD / segmentation config
VAD_SILENCE_MS = 500  # ms of silence before we consider speech done
VAD_MIN_SPEECH_MS = 400  # ignore speech segments shorter than this
MAX_SEGMENT_S = 10  # hard cap — transcribe even if no silence detected
# -----------------------

# Wake word gating
CTRL_PORT = 12346
BLEED_SKIP_PACKETS = (
    8  # ~256ms of audio to discard after wake word (covers "Elio" utterance bleed)
)

listen_state = ListenState.IDLE
bleed_remaining = 0
state_lock = threading.Lock()

packet_queue: collections.deque = collections.deque()
vad_queue: collections.deque = collections.deque()
queue_lock = threading.Lock()
leftover: np.ndarray = np.zeros(0, dtype=np.float32)

# A separate queue to pass completed audio segments to the transcription thread
transcribe_queue: queue.Queue = queue.Queue()

# --- VAD accumulator state ---
accumulator: list[np.ndarray] = []
silence_packets = 0
SILENCE_PACKETS_MAX = int((VAD_SILENCE_MS / 1000) * SAMPLE_RATE / SAMPLES_PER_PKT)
MIN_SPEECH_PACKETS = int((VAD_MIN_SPEECH_MS / 1000) * SAMPLE_RATE / SAMPLES_PER_PKT)
MAX_SEGMENT_PACKETS = int(MAX_SEGMENT_S * SAMPLE_RATE / SAMPLES_PER_PKT)


def control_listener() -> None:
    """Listens on CTRL_PORT for wake word trigger packets from the ESP32."""
    global listen_state, bleed_remaining

    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_sock.bind(("0.0.0.0", CTRL_PORT))
    print(f"{ts()} Control listener ready on port {CTRL_PORT}")

    while True:
        data, addr = ctrl_sock.recvfrom(16)
        if not data or data[0] != 0x01:
            continue

        with state_lock:
            if listen_state != ListenState.IDLE:
                print(
                    f"{ts()} [CTRL] Wake signal received but state is {listen_state.name}, ignoring."
                )
                continue
            listen_state = ListenState.SKIP_WAKEWORD_BLEED
            bleed_remaining = BLEED_SKIP_PACKETS

        print(
            f"\n{ts()} [WAKE] Wake word received from {addr[0]}! Skipping {BLEED_SKIP_PACKETS} packets of bleed..."
        )


def vad_accumulator_loop() -> None:
    global accumulator, silence_packets, listen_state, bleed_remaining

    while True:
        chunk = None
        with queue_lock:
            if vad_queue:
                chunk = vad_queue.popleft()

        if chunk is None:
            time.sleep(0.005)
            continue

        with state_lock:
            current_state = listen_state

        # --- IDLE: discard everything ---
        if current_state == ListenState.IDLE:
            continue

        # --- SKIP_WAKEWORD_BLEED: count down and discard ---
        if current_state == ListenState.SKIP_WAKEWORD_BLEED:
            with state_lock:
                bleed_remaining -= 1
                if bleed_remaining <= 0:
                    listen_state = ListenState.CAPTURING
                    accumulator = []
                    silence_packets = 0
                    print(f"{ts()} [WAKE] Bleed skip done. Capturing command now...")
            continue

        # --- TRANSCRIBING: don't accumulate while Whisper is busy ---
        if current_state == ListenState.TRANSCRIBING:
            continue

        # --- CAPTURING: normal VAD logic ---
        rms = float(np.sqrt(np.mean(chunk**2)))
        is_speech = rms >= VAD_SILENCE_THRESHOLD

        if rms > 0.001:
            sys.stdout.write(
                f"\rVAD RMS={rms:.4f} speech={is_speech} acc_len={len(accumulator)} "
            )
            sys.stdout.flush()

        if is_speech:
            accumulator.append(chunk)
            silence_packets = 0
        else:
            if accumulator:
                silence_packets += 1
                accumulator.append(chunk)

        if accumulator:
            end_of_speech = (not is_speech) and (silence_packets >= SILENCE_PACKETS_MAX)
            too_long = len(accumulator) >= MAX_SEGMENT_PACKETS

            if end_of_speech or too_long:
                segment = np.concatenate(accumulator)
                if len(accumulator) >= MIN_SPEECH_PACKETS:
                    print(
                        f"\n{ts()} [VAD] Segment ready: {len(accumulator)} packets, {len(segment)} samples"
                    )
                    with state_lock:
                        listen_state = ListenState.TRANSCRIBING
                    transcribe_queue.put(segment)
                else:
                    print(
                        f"\n{ts()} [VAD] Segment too short ({len(accumulator)} pkts), discarding"
                    )
                    with state_lock:
                        listen_state = ListenState.IDLE
                accumulator = []
                silence_packets = 0


def transcription_loop(model: WhisperModel) -> None:
    global listen_state

    while True:
        segment: np.ndarray = transcribe_queue.get()

        try:
            print(
                f"{ts()} [transcribe] Got segment of {len(segment)} samples, transcribing...",
                flush=True,
            )
            segments, info = model.transcribe(
                segment,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                word_timestamps=False,
            )

            text = " ".join(s.text.strip() for s in segments).strip()
            if text:
                print(f"{ts()} [transcript] {text}")
        except Exception as exc:
            print(f"{ts()} [transcribe error] {exc}", flush=True)
        finally:
            # Always reset to IDLE so the next wake word can be accepted
            with state_lock:
                listen_state = ListenState.IDLE
            print(f"{ts()} [STATE] Ready. Waiting for wake word...")


def receive_loop(sock: socket.socket) -> None:
    """Background thread: receive UDP packets and enqueue decoded audio."""
    expected_bytes = SAMPLES_PER_PKT * 2  # uint16 = 2 bytes each
    while True:
        data, _ = sock.recvfrom(expected_bytes * 2)
        if len(data) != expected_bytes:
            continue
        raw = np.frombuffer(data, dtype="<u2").astype(np.float32)
        audio = (raw - 2048.0) / 2048.0
        audio = audio - np.mean(audio)

        # Noise gate: only applies to playback if you want to suppress idle hiss.
        # For full-fidelity recording/monitoring, set NOISE_GATE = 0 or remove this block.
        if NOISE_GATE > 0 and np.sqrt(np.mean(audio**2)) < NOISE_GATE:
            playback_audio = np.zeros(SAMPLES_PER_PKT, dtype=np.float32)
        else:
            playback_audio = audio

        with queue_lock:
            if len(packet_queue) >= MAX_QUEUE_LEN:
                packet_queue.popleft()
            packet_queue.append(playback_audio)  # <-- was using gated audio
            vad_queue.append(audio)
            if len(vad_queue) > MAX_QUEUE_LEN * 4:
                vad_queue.popleft()


def audio_callback(outdata: np.ndarray, frames: int, time, status) -> None:
    """sounddevice callback: fill outdata with queued audio, silence on underrun."""
    global leftover

    output = np.zeros(frames, dtype=np.float32)
    write_pos = 0
    needed = frames

    # Use leftover samples from previous callback first
    if len(leftover) > 0:
        use = min(len(leftover), needed)
        output[write_pos : write_pos + use] = leftover[:use]
        leftover = leftover[use:]
        write_pos += use
        needed -= use

    # Pull packets from queue until we have enough samples
    while needed > 0:
        with queue_lock:
            if not packet_queue:
                break  # note: should be break, not continue
            chunk = packet_queue.popleft()
        if len(chunk) <= needed:
            output[write_pos : write_pos + len(chunk)] = chunk
            write_pos += len(chunk)
            needed -= len(chunk)
        else:
            # Chunk is larger than remaining space — save the tail for next callback
            output[write_pos : write_pos + needed] = chunk[:needed]
            leftover = chunk[needed:]
            needed = 0

    outdata[:, 0] = output


def main() -> None:
    print(f"{ts()} Loading Whisper model '{WHISPER_MODEL}'...")
    model = WhisperModel(
        WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE
    )
    print(f"{ts()} Model loaded.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"{ts()} Listening for UDP audio on port {UDP_PORT}...")

    # Start all background threads
    for target, args in [
        (receive_loop, (sock,)),
        (control_listener, ()),
        (vad_accumulator_loop, ()),
        (transcription_loop, (model,)),
    ]:
        t = threading.Thread(target=target, args=args, daemon=True)
        t.start()

    print(f"{ts()} Waiting for {PREBUFFER_PKTS} packets to pre-buffer...")
    while True:
        with queue_lock:
            if len(packet_queue) >= PREBUFFER_PKTS:
                break

    print(f"{ts()} Starting playback. Press Ctrl+C to stop.")
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=SAMPLES_PER_PKT,
    ):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print(f"\n{ts()} Stopped.")


if __name__ == "__main__":
    main()
