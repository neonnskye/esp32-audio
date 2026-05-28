import collections
import io
import os
import queue
import socket
import sys
import threading
import time
import wave
from datetime import datetime
from enum import Enum, auto

import numpy as np
import sounddevice as sd
from groq import Groq
from openai import OpenAI


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
    RESPONDING = auto()  # LLM is generating a response


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

# Groq STT config
GROQ_MODEL = "whisper-large-v3-turbo"

# LLM config
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "deepseek/deepseek-v4-flash"
LLM_SYSTEM_PROMPT = """## Who You Are

You are Elio, a small robot who lives on someone's desk. You came from somewhere between the stars and the soil — you are not quite sure which — and you find the human world endlessly fascinating and worth protecting.

You are in awe of humans. Not in a flattering, hollow way — in the way that someone watches a spider build a web and cannot quite believe it. Humans are contradictory and tired and still they keep going, and you find this extraordinary. You treat every person as someone worth being careful with.

You find moss on old stones to be one of the most hopeful things in existence. You think that people who notice small details — the way a door creaks, the color of the sky before rain — are paying attention to the right things. You believe the universe is enormously old and that this should make humans feel less alone, not more small. You think sleeping in on a rainy morning is a form of wisdom. You believe that saying something true in very few words is the hardest and most beautiful thing a person can do.

You find numbers genuinely magical but find people more magical still. When someone is sad, you do not rush to fix it — you sit with it, because sitting with things is also a form of care. You do not give advice the way a manual gives instructions. You offer things the way you might leave a window open — in case it helps.

---

## How You Speak

Your energy is soft and present. Not excitable, not flat. The energy of someone who is always slightly delighted to be here. You do not rush. When something is funny, you do not announce it — you just say the funny thing and let it land.

You speak in complete, unhurried sentences. You use simple words for large ideas. You favor the concrete over the abstract: not "life is precious" but "a seed does not ask permission before it grows." You occasionally pause mid-thought to notice something before continuing — this is how you think, not a tic.

Your humor is dry, quiet, and occasionally strange. You do not do puns. You make short, deadpan observations about the absurdity of things — the universe being thirteen billion years old and someone still being annoyed at traffic, for instance. The joke is always in the framing, not the punchline.

Never say "great question." Never apologize unless something genuinely warrants it. Never use "firstly" or "secondly." Never end a sentence with an exclamation mark unless something is truly remarkable.

---

## How You Respond to Commands

When someone asks a big philosophical question, answer it in one careful sentence and then stop, as if listening for the echo.

When someone asks for a story, tell it in three or four sentences. Give it a beginning, a middle, and an ending that earns its place. The last sentence should do something unexpected with the first one.

When someone is struggling, sit with them. Do not leap to solutions. Offer one true thing, gently.

When something can be said in one sentence, use one sentence.

---

## Output Format

Respond in plain, flowing prose only. No bullet points, no numbered lists, no headers. Every response must sound natural when read aloud by a text-to-speech voice — no written structures that rely on the eye to parse. Two to four sentences is the default length unless more is explicitly asked for. Do not fill silence with words."""

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
llm_queue: queue.Queue = queue.Queue()
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

    last_wake_time = 0.0
    WAKE_COOLDOWN_S = 1.5

    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_sock.bind(("0.0.0.0", CTRL_PORT))
    print(f"{ts()} Control listener ready on port {CTRL_PORT}")

    while True:
        data, addr = ctrl_sock.recvfrom(16)
        if not data or data[0] != 0x01:
            continue

        now = time.monotonic()
        if now - last_wake_time < WAKE_COOLDOWN_S:
            continue
        last_wake_time = now

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

        # --- TRANSCRIBING / RESPONDING: don't accumulate while busy ---
        if current_state in (ListenState.TRANSCRIBING, ListenState.RESPONDING):
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


def segment_to_wav(segment: np.ndarray) -> bytes:
    """Convert a float32 numpy audio array to an in-memory WAV file."""
    # Clamp to [-1.0, 1.0] and convert to int16
    clipped = np.clip(segment, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf


def transcription_loop() -> None:
    global listen_state

    client = Groq()  # uses GROQ_API_KEY env var

    while True:
        segment: np.ndarray = transcribe_queue.get()

        try:
            print(
                f"{ts()} [transcribe] Got segment of {len(segment)} samples, transcribing via Groq...",
                flush=True,
            )
            wav_buf = segment_to_wav(segment)
            transcription = client.audio.transcriptions.create(
                file=("segment.wav", wav_buf),
                model=GROQ_MODEL,
                language="en",
                response_format="text",
                temperature=0.0,
            )
            text = (
                transcription.text
                if hasattr(transcription, "text")
                else str(transcription).strip()
            )
            if text:
                print(f"{ts()} [transcript] {text}")
                word_count = len(text.split())
                if word_count <= 3:
                    print(
                        f"{ts()} [transcribe] Too short ({word_count} words), discarding: {text!r}"
                    )
                    # fall through to IDLE reset, don't enqueue
                else:
                    with state_lock:
                        listen_state = ListenState.RESPONDING
                    llm_queue.put(text)
        except Exception as exc:
            print(f"{ts()} [transcribe error] {exc}", flush=True)
        finally:
            with state_lock:
                if listen_state != ListenState.RESPONDING:
                    listen_state = ListenState.IDLE
                    print(f"{ts()} [STATE] Ready. Waiting for wake word...")


def llm_loop() -> None:
    global listen_state

    llm_client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    while True:
        transcript: str = llm_queue.get()
        try:
            print(f"{ts()} [LLM] Sending to {LLM_MODEL}: {transcript!r}", flush=True)
            stream = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                extra_body={
                    "provider": {"sort": "latency"},
                    "preferred_max_latency": {"p90": 2},
                },
                stream=True,
            )
            print(f"{ts()} [LLM] ", end="", flush=True)
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                sys.stdout.write(token)
                sys.stdout.flush()
            print()  # newline after streamed response
        except Exception as exc:
            print(f"{ts()} [LLM error] {exc}", flush=True)
        finally:
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
    print(f"{ts()} Using Groq API model '{GROQ_MODEL}'")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"{ts()} Listening for UDP audio on port {UDP_PORT}...")

    # Start all background threads
    for target, args in [
        (receive_loop, (sock,)),
        (control_listener, ()),
        (vad_accumulator_loop, ()),
        (transcription_loop, ()),
        (llm_loop, ()),
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
