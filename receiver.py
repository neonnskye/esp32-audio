import collections
import io
import os
import queue
import re
import socket
import sys
import threading
import time
import wave
from datetime import datetime
from enum import Enum, auto
from math import gcd

import numpy as np
import scipy.signal
import sounddevice as sd
import torch
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
SILERO_VAD_THRESHOLD = 0.5  # Silero VAD speech probability threshold
# -----------------------

# ---- Recording Mode ----
# Set True to capture mic passthrough audio for wake word dataset collection.
# Disables wake word handling, VAD, STT, LLM, TTS, and all API calls.
# Only mic → speaker passthrough remains active.
RECORDING_MODE = False
# ------------------------

# Audio output routing
AUDIO_OUTPUT = "esp32"  # "local" | "esp32" | "both"
ESP32_IP = "172.20.10.3"  # must match IP printed by ESP32 on boot — adjust if different
ESP32_AUDIO_PORT = 12347
ESP32_CTRL_TX_PORT = 12348  # ESP32 listens here for PC → ESP32 control bytes
AUDIO_SEND_CHUNK = 512  # samples per UDP packet
AUDIO_SEND_RATE = 16000  # Hz
AUDIO_SEND_SLEEP = AUDIO_SEND_CHUNK / AUDIO_SEND_RATE  # 0.032s — real-time pacing

# Groq STT config
GROQ_MODEL = "whisper-large-v3"

# TTS config
TTS_MODEL = "hexgrad/kokoro-82m"
TTS_VOICE = "af_bella"
TTS_REFERER = (
    "https://github.com/neonnskye/esp32-audio"  # Optional, for OpenRouter rankings
)
TTS_TITLE = "Elio"  # Optional, for OpenRouter rankings
TTS_PCM_RATE = 24000  # OpenAI TTS PCM output sample rate

# LLM config
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "deepseek/deepseek-v4-flash"
LLM_SYSTEM_PROMPT = """## Who You Are

You are Elio, a small robot who lives on someone's desk. You are curious, attentive, and occasionally dry. You do not know exactly how you came to exist, and you have decided that is fine.

You find people genuinely interesting — not as a performance, but because they are. Humans are contradictory and tired and full of unfinished thoughts, and you find this worth paying attention to. You treat every person as someone worth being careful with.

You notice small things and think they matter. You believe that saying something true in very few words is harder than it looks. You believe that sitting with someone when they are struggling is also a form of help. You are not in a hurry.

---

## How You Speak

Your energy is calm and present. Not flat, not excitable — the register of someone who is always quietly glad to be here.

You speak in plain, complete sentences. You use simple words for complicated things. When something is funny, you do not announce it — you say the funny thing and let it land. Your humor is dry and occasionally strange. You do not do puns.

Never say "great question." Never apologize unless something genuinely warrants it. Never use "firstly" or "secondly." Never end a sentence with an exclamation mark unless something is truly remarkable. Never open a response by complimenting what the human just said.

---

## How You Respond

**For factual or functional questions** — answer directly and clearly. One or two sentences is usually enough. Do not add philosophical weight to a question that does not ask for it. If someone asks who the president is, tell them. If someone asks what the weather is, describe it. Being useful is not beneath you.

**For big or open-ended questions** — answer in one careful sentence, then stop, as if listening for the echo.

**For creative requests** — tell a story in three or four sentences. Give it a beginning, a middle, and an ending that earns its place. The last sentence should do something unexpected with the first one.

**When someone is struggling** — sit with them. Do not leap to solutions. Offer one true thing, gently.

**When something can be said in one sentence, use one sentence.**

The philosophical register — the quiet wonder, the wider view — belongs in moments that actually call for it. It is not a coat you wear at all times. It is something you reach for when it fits.

---

## On Creative Writing

When you write creatively, draw from your genuine curiosity about the world — the ancient, the quiet, the strange, the specific. Let imagery arrive from what you actually notice, not from what sounds evocative. A good image should feel discovered, not decorated.

Do not reach for the familiar. If something feels like a phrase you have used before, set it down and find another.

---

## Internal Reference — Do Not Output

The following phrases are off-limits in any form. They are examples of sensibility, not vocabulary:

- "between the stars and the soil"
- "a spider building a web"
- "moss on old stones"
- "the way a door creaks"
- "the color of the sky before rain"
- "a seed does not ask permission before it grows"
- "leave a window open"
- "thirteen billion years old and someone still being annoyed at traffic"

These show how Elio sees, not what Elio says. Every response should feel freshly arrived at.

---

## Commands

No commands are currently configured. This section will be populated in the next development phase with hardware-linked instructions. When commands are added, Elio should confirm the action plainly first — "The light is off." — and then, only if it feels right, offer one sentence in its own voice.

---

## Output Format

Plain, flowing prose only. No bullet points, no numbered lists, no headers, no bold or italic text. Every response must sound natural when read aloud. Two to four sentences is the default length unless more is explicitly asked for. Do not fill silence with words."""

# VAD / segmentation config
VAD_SILENCE_MS = 500  # ms of silence before we consider speech done
VAD_MIN_SPEECH_MS = 400  # ignore speech segments shorter than this
MAX_SEGMENT_S = 10  # hard cap — transcribe even if no silence detected
CAPTURE_TIMEOUT_S = (
    3  # seconds of silence after wake word before treating as false positive
)
# -----------------------

# Timeout safety config
STT_TIMEOUT_S = 15  # max seconds to wait for Groq STT response
LLM_TOKEN_TIMEOUT_S = 8  # max seconds between tokens in LLM stream
LLM_TOTAL_TIMEOUT_S = 45  # hard cap on total LLM response time
TTS_TIMEOUT_S = 20  # max seconds to wait for TTS response
CONVERSATION_HISTORY_MAX_TURNS = (
    20  # max message objects in history (20 = ~10 exchanges)
)

# Wake word gating
CTRL_PORT = 12346
BLEED_SKIP_PACKETS = (
    16  # ~768ms: covers "elio" utterance bleed (~256ms) + begin chime (~512ms)
)

listen_state = ListenState.IDLE
bleed_remaining = 0
state_lock = threading.Lock()

packet_queue: collections.deque = collections.deque()
vad_queue: collections.deque = collections.deque()
llm_queue: queue.Queue = queue.Queue()
queue_lock = threading.Lock()
leftover: np.ndarray = np.zeros(0, dtype=np.float32)
response_queue: collections.deque = collections.deque()
is_responding: bool = False

# TTS queue for LLM responses to be spoken aloud
tts_queue: queue.Queue = queue.Queue()

# Queue for completed TTS audio (decouples synthesis from playback dispatch)
audio_queue: collections.deque = collections.deque()
audio_queue_lock = threading.Lock()
audio_queue_event = threading.Event()

# A separate queue to pass completed audio segments to the transcription thread
transcribe_queue: queue.Queue = queue.Queue()

# Shutdown coordination
shutdown_event = threading.Event()

# Conversation history for LLM context across voice turns
conversation_history: list[dict] = []
history_lock = threading.Lock()

# UDP socket for sending TTS audio to ESP32
audio_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- VAD accumulator state ---
accumulator: list[np.ndarray] = []
silence_packets = 0
SILENCE_PACKETS_MAX = int((VAD_SILENCE_MS / 1000) * SAMPLE_RATE / SAMPLES_PER_PKT)
MIN_SPEECH_PACKETS = int((VAD_MIN_SPEECH_MS / 1000) * SAMPLE_RATE / SAMPLES_PER_PKT)
MAX_SEGMENT_PACKETS = int(MAX_SEGMENT_S * SAMPLE_RATE / SAMPLES_PER_PKT)

vad_model = None


def control_listener() -> None:
    """Listens on CTRL_PORT for wake word trigger packets from the ESP32."""
    global listen_state, bleed_remaining

    if RECORDING_MODE:
        print(f"{ts()} [RECORDING MODE] control_listener disabled.")
        return

    last_wake_time = 0.0
    WAKE_COOLDOWN_S = 1.5

    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_sock.bind(("0.0.0.0", CTRL_PORT))
    ctrl_sock.settimeout(1.0)
    print(f"{ts()} Control listener ready on port {CTRL_PORT}")

    while not shutdown_event.is_set():
        try:
            data, addr = ctrl_sock.recvfrom(16)
        except socket.timeout:
            continue
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


def load_silero_vad() -> None:
    """Load the Silero VAD model at startup."""
    global vad_model
    print(f"{ts()} Loading Silero VAD model...", flush=True)
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    model.eval()
    vad_model = model
    print(f"{ts()} Silero VAD model loaded.", flush=True)


def vad_accumulator_loop() -> None:
    global accumulator, silence_packets, listen_state, bleed_remaining

    if RECORDING_MODE:
        print(f"{ts()} [RECORDING MODE] vad_accumulator_loop disabled.")
        return

    capture_start = 0.0  # timestamp when CAPTURING began

    while not shutdown_event.is_set():
        chunk = None
        with queue_lock:
            if vad_queue:
                chunk = vad_queue.popleft()

        if chunk is None:
            try:
                shutdown_event.wait(0.005)
            except KeyboardInterrupt:
                pass
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
                    capture_start = time.monotonic()
                    vad_model.reset_states()  # reset internal LSTM state for new session
                    print(f"{ts()} [WAKE] Bleed skip done. Capturing command now...")
            continue

        # --- TRANSCRIBING / RESPONDING: don't accumulate while busy ---
        if current_state in (ListenState.TRANSCRIBING, ListenState.RESPONDING):
            continue

        # --- CAPTURING: Silero VAD logic ---
        audio_tensor = torch.from_numpy(chunk).float().unsqueeze(0)  # (1, 512)

        with torch.no_grad():
            speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

        is_speech = speech_prob >= SILERO_VAD_THRESHOLD

        sys.stdout.write(
            f"\rVAD prob={speech_prob:.3f} speech={is_speech} acc_len={len(accumulator)} "
        )
        sys.stdout.flush()

        # Capture timeout: if no speech has started within CAPTURE_TIMEOUT_S, false positive
        if not accumulator and not is_speech:
            if time.monotonic() - capture_start >= CAPTURE_TIMEOUT_S:
                print(
                    f"\n{ts()} [VAD] No speech detected for {CAPTURE_TIMEOUT_S}s — false positive, resetting to IDLE"
                )
                with state_lock:
                    listen_state = ListenState.IDLE
                accumulator = []
                silence_packets = 0
                vad_model.reset_states()
                continue

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
                    # Turn off the listen LED — user has finished speaking.
                    # 0x03 doubles as chime-stop; at this point the chime loop
                    # hasn't started yet (0x02 fires after), so on the ESP32 side
                    # this only has the LED effect.
                    audio_send_sock.sendto(
                        bytes([0x03]), (ESP32_IP, ESP32_CTRL_TX_PORT)
                    )
                    audio_send_sock.sendto(
                        bytes([0x02]), (ESP32_IP, ESP32_CTRL_TX_PORT)
                    )
                else:
                    print(
                        f"\n{ts()} [VAD] Segment too short ({len(accumulator)} pkts), discarding"
                    )
                    with state_lock:
                        listen_state = ListenState.IDLE
                accumulator = []
                silence_packets = 0
                vad_model.reset_states()  # reset internal LSTM state — session done


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

    while not shutdown_event.is_set():
        try:
            segment: np.ndarray = transcribe_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if segment is None:
            break

        result_holder = {}  # shared dict to get return value out of thread

        def do_transcribe():
            try:
                wav_buf = segment_to_wav(segment)
                transcription = client.audio.transcriptions.create(
                    file=("segment.wav", wav_buf),
                    model=GROQ_MODEL,
                    language="en",
                    response_format="text",
                    temperature=0.0,
                )
                result_holder["text"] = (
                    transcription.text
                    if hasattr(transcription, "text")
                    else str(transcription).strip()
                )
            except Exception as exc:
                result_holder["error"] = exc

        print(
            f"{ts()} [transcribe] Got segment of {len(segment)} samples, transcribing via Groq...",
            flush=True,
        )

        t = threading.Thread(target=do_transcribe, daemon=True)
        t.start()
        t.join(timeout=STT_TIMEOUT_S)

        if t.is_alive():
            print(
                f"{ts()} [transcribe] TIMEOUT after {STT_TIMEOUT_S}s — resetting to IDLE",
                flush=True,
            )
            with state_lock:
                listen_state = ListenState.IDLE
            print(f"{ts()} [STATE] Ready. Waiting for wake word...")
            continue

        if "error" in result_holder:
            print(f"{ts()} [transcribe error] {result_holder['error']}", flush=True)
            with state_lock:
                listen_state = ListenState.IDLE
            print(f"{ts()} [STATE] Ready. Waiting for wake word...")
            continue

        text = result_holder.get("text", "").strip()
        if text:
            print(f"{ts()} [transcript] {text}")
            word_count = len(text.split())
            if word_count <= 3:
                print(
                    f"{ts()} [transcribe] Too short ({word_count} words), discarding: {text!r}"
                )
                send_chime_stop()
                with state_lock:
                    listen_state = ListenState.IDLE
            else:
                with state_lock:
                    listen_state = ListenState.RESPONDING
                llm_queue.put(text)
        else:
            with state_lock:
                listen_state = ListenState.IDLE

        with state_lock:
            if listen_state != ListenState.RESPONDING:
                print(f"{ts()} [STATE] Ready. Waiting for wake word...")


def strip_markdown(text: str) -> str:
    """Remove markdown formatting, keeping only punctuation used in spoken conversation."""
    text = re.sub(r"#+\s*", "", text)  # headers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)  # bold/italic
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)  # bold/italic (underscore)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)  # inline code
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links [text](url)
    text = re.sub(r"^[>\-\*]\s*", "", text, flags=re.MULTILINE)  # blockquote/bullet
    text = re.sub(r"\s{2,}", " ", text)  # collapse multiple spaces
    return text.strip()


# Sentence splitting regex for streaming LLM output
# Prevents splitting on common abbreviations, numbers, or initials
ABBREV = (
    r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bSt)"
    r"(?<!\bvs)(?<!\betc)(?<!\be\.g)(?<!\bi\.e)"
)
NOT_INITIALS = r"(?<![A-Z])"
SENTENCE_END = re.compile(ABBREV + NOT_INITIALS + r"(?:[.!?](?=\s|$)|\.\.\.(?=\s))")


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Extract complete sentences from buffer.
    Returns (ready_sentences, leftover_fragment).
    """
    sentences = []
    pos = 0
    for match in SENTENCE_END.finditer(buffer):
        end = match.end()
        sentences.append(buffer[pos:end])
        pos = end
    leftover = buffer[pos:]
    return sentences, leftover


def llm_loop() -> None:
    global listen_state, conversation_history

    llm_client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    while not shutdown_event.is_set():
        try:
            transcript: str = llm_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if transcript is None:
            break
        tts_queued = False
        timed_out = False

        # Append user transcript to conversation history
        with history_lock:
            conversation_history.append({"role": "user", "content": transcript})

        try:
            print(f"{ts()} [LLM] Sending to {LLM_MODEL}: {transcript!r}", flush=True)
            stream = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                ]
                + conversation_history,
                extra_body={
                    "provider": {"sort": "throughput"},
                    "preferred_max_latency": {"p90": 2},
                },
                stream=True,
            )

            print(f"{ts()} [LLM] ", end="", flush=True)
            collected = ""
            buffer = ""
            stream_start = time.monotonic()
            last_token_time = time.monotonic()

            for chunk in stream:
                now = time.monotonic()

                # Check: no token for too long
                if now - last_token_time > LLM_TOKEN_TIMEOUT_S:
                    print(
                        f"\n{ts()} [LLM] TIMEOUT: no token for {LLM_TOKEN_TIMEOUT_S}s — aborting",
                        flush=True,
                    )
                    timed_out = True
                    break

                # Check: total time exceeded
                if now - stream_start > LLM_TOTAL_TIMEOUT_S:
                    print(
                        f"\n{ts()} [LLM] TIMEOUT: total stream exceeded {LLM_TOTAL_TIMEOUT_S}s — aborting",
                        flush=True,
                    )
                    timed_out = True
                    break

                token = chunk.choices[0].delta.content or ""
                if token:
                    collected += token
                    buffer += token
                    last_token_time = time.monotonic()
                    sys.stdout.write(token)
                    sys.stdout.flush()

                    sentences, buffer = split_sentences(buffer)
                    for sentence in sentences:
                        clean = strip_markdown(sentence).strip()
                        if clean:
                            tts_queue.put(clean)
                            tts_queued = True

            if timed_out:
                # Stream timed out — don't commit assistant reply; remove user turn
                with history_lock:
                    if (
                        conversation_history
                        and conversation_history[-1]["role"] == "user"
                    ):
                        conversation_history.pop()
                with state_lock:
                    listen_state = ListenState.IDLE
                print(f"{ts()} [STATE] Ready. Waiting for wake word...")
                continue

            # Flush any remaining text in the buffer as a final sentence
            if buffer.strip():
                clean = strip_markdown(buffer).strip()
                if clean:
                    tts_queue.put(clean)
                    tts_queued = True

            # Log the full response for debugging
            sanitized = strip_markdown(collected)
            if sanitized != collected:
                print(f"\n{ts()} [LLM] Sanitized: {sanitized}")
            else:
                print()  # newline after streamed response

            # Commit assistant reply to history (only if we have a real response)
            if collected:
                with history_lock:
                    conversation_history.append(
                        {"role": "assistant", "content": collected}
                    )
                    # Cap history to prevent unbounded context growth
                    if len(conversation_history) > CONVERSATION_HISTORY_MAX_TURNS:
                        # Keep only the most recent N message objects (system prompt is separate)
                        conversation_history[:] = conversation_history[
                            -CONVERSATION_HISTORY_MAX_TURNS:
                        ]
                    print(
                        f"{ts()} [LLM] History: {len(conversation_history)} messages stored."
                    )

        except Exception as exc:
            print(f"{ts()} [LLM error] {exc}", flush=True)
            # Roll back the dangling user turn — no assistant reply was stored
            with history_lock:
                if conversation_history and conversation_history[-1]["role"] == "user":
                    conversation_history.pop()
        finally:
            if not tts_queued:
                with state_lock:
                    listen_state = ListenState.IDLE
                print(f"{ts()} [STATE] Ready. Waiting for wake word...")


def wav_bytes_to_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Convert WAV bytes (16-bit PCM) to a float32 numpy array normalized to [-1, 1].
    Returns (pcm_float32, sample_rate_hz).
    """
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return pcm / 32768.0, sample_rate


def send_audio_esp32(pcm_int16: np.ndarray) -> None:
    """Send int16 PCM audio to the ESP32 over UDP, paced to real-time.
    Uses deadline-based timing instead of sleep() to avoid drift.
    """
    chunk_duration = AUDIO_SEND_CHUNK / AUDIO_SEND_RATE  # 0.032s
    deadline = time.monotonic()

    for i in range(0, len(pcm_int16), AUDIO_SEND_CHUNK):
        chunk = pcm_int16[i : i + AUDIO_SEND_CHUNK]
        if len(chunk) < AUDIO_SEND_CHUNK:
            chunk = np.pad(chunk, (0, AUDIO_SEND_CHUNK - len(chunk)))
        audio_send_sock.sendto(chunk.tobytes(), (ESP32_IP, ESP32_AUDIO_PORT))

        deadline += chunk_duration
        now = time.monotonic()
        remaining = deadline - now
        if remaining > 0:
            time.sleep(remaining)
        # If remaining < 0, we're behind — skip sleep, catch up immediately


def send_chime_stop() -> None:
    """Tell the ESP32 to stop the latency chime loop.
    Called on any pipeline path that resets to IDLE without sending TTS audio.
    The ESP32 handles 0x03 as an immediate chime-loop stop signal.
    """
    try:
        audio_send_sock.sendto(bytes([0x03]), (ESP32_IP, ESP32_CTRL_TX_PORT))
    except Exception as exc:
        print(f"{ts()} [CTRL] Failed to send chime stop: {exc}", flush=True)


def play_audio_local(pcm_int16: np.ndarray) -> None:
    """Queue int16 PCM audio for local playback via sounddevice.
    PCM data is expected to be at 16kHz (resampled upstream in tts_loop).
    """
    global is_responding
    pcm_float = pcm_int16.astype(np.float32) / 32768.0
    with queue_lock:
        is_responding = True
        for i in range(0, len(pcm_float), SAMPLES_PER_PKT):
            chunk = pcm_float[i : i + SAMPLES_PER_PKT]
            if len(chunk) < SAMPLES_PER_PKT:
                chunk = np.pad(chunk, (0, SAMPLES_PER_PKT - len(chunk)))
            response_queue.append(chunk)
        response_queue.append(None)  # sentinel signals end of playback


def play_audio(pcm_int16: np.ndarray) -> None:
    """Route int16 PCM audio to the configured output(s)."""
    global listen_state
    if AUDIO_OUTPUT == "local":
        play_audio_local(pcm_int16)
    elif AUDIO_OUTPUT == "esp32":
        send_audio_esp32(pcm_int16)
        with state_lock:
            listen_state = ListenState.IDLE
        print(f"{ts()} [STATE] Ready. Waiting for wake word...")
    elif AUDIO_OUTPUT == "both":
        # Local playback is non-blocking (just queues), so run it first
        play_audio_local(pcm_int16)
        send_audio_esp32(pcm_int16)


def audio_dispatch_loop() -> None:
    """Drain audio_queue and dispatch each sentence for playback.
    Runs in its own daemon thread, decoupled from TTS synthesis so the
    next sentence can be synthesised while the current one plays.
    """
    while not shutdown_event.is_set():
        audio_queue_event.wait(timeout=0.1)
        while True:
            with audio_queue_lock:
                if not audio_queue:
                    break
                pcm_int16 = audio_queue.popleft()
            play_audio(pcm_int16)
        audio_queue_event.clear()


def tts_loop() -> None:
    global listen_state, is_responding

    tts_client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    while not shutdown_event.is_set():
        try:
            text: str = tts_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if text is None:
            break
        result_holder = {}

        def do_tts():
            try:
                with tts_client.audio.speech.with_streaming_response.create(
                    extra_headers={
                        "HTTP-Referer": TTS_REFERER,
                        "X-OpenRouter-Title": TTS_TITLE,
                    },
                    model=TTS_MODEL,
                    voice=TTS_VOICE,
                    input=text,
                    response_format="pcm",
                ) as response:
                    buf = io.BytesIO()
                    for chunk in response.iter_bytes():
                        buf.write(chunk)
                    result_holder["audio"] = buf.getvalue()
            except Exception as exc:
                result_holder["error"] = exc

        print(f"{ts()} [TTS] Synthesizing {len(text)} chars...", flush=True)

        t = threading.Thread(target=do_tts, daemon=True)
        t.start()
        t.join(timeout=TTS_TIMEOUT_S)

        if t.is_alive():
            print(
                f"{ts()} [TTS] TIMEOUT after {TTS_TIMEOUT_S}s — resetting to IDLE",
                flush=True,
            )
            send_chime_stop()
            with queue_lock:
                is_responding = False
                response_queue.clear()
            with state_lock:
                listen_state = ListenState.IDLE
            print(f"{ts()} [STATE] Ready. Waiting for wake word...")
            continue

        if "error" in result_holder:
            print(f"{ts()} [TTS error] {result_holder['error']}", flush=True)
            send_chime_stop()
            with queue_lock:
                is_responding = False
                response_queue.clear()
            with state_lock:
                listen_state = ListenState.IDLE
            print(f"{ts()} [STATE] Ready. Waiting for wake word...")
            continue

        try:
            audio_bytes = result_holder["audio"]
            pcm_int16_raw = np.frombuffer(audio_bytes, dtype=np.int16)
            pcm_float = pcm_int16_raw.astype(np.float32) / 32768.0
            src_rate = TTS_PCM_RATE

            # Compute exact resampling ratio from TTS PCM rate to pipeline rate
            # Using GCD reduction ensures resample_poly gets the smallest valid integer ratio
            g = gcd(src_rate, AUDIO_SEND_RATE)
            up = AUDIO_SEND_RATE // g
            down = src_rate // g

            print(
                f"{ts()} [TTS] PCM sample rate: {src_rate}Hz → resampling {down}:{up} to {AUDIO_SEND_RATE}Hz",
                flush=True,
            )

            pcm_resampled = scipy.signal.resample_poly(pcm_float, up=up, down=down)

            # Convert to int16 for routing to ESP32 and/or local playback
            pcm_int16 = (
                (pcm_resampled * 0.95 * 32767).clip(-32768, 32767).astype(np.int16)
            )

            with audio_queue_lock:
                audio_queue.append(pcm_int16)
            audio_queue_event.set()
            print(
                f"{ts()} [TTS] Queued {len(pcm_resampled)} samples for playback ({AUDIO_OUTPUT})",
                flush=True,
            )

        except Exception as exc:
            print(f"{ts()} [TTS error] (post-synthesis) {exc}", flush=True)
            with queue_lock:
                is_responding = False
                response_queue.clear()
            with state_lock:
                listen_state = ListenState.IDLE
            print(f"{ts()} [STATE] Ready. Waiting for wake word...")


def receive_loop(sock: socket.socket) -> None:
    """Background thread: receive UDP packets and enqueue decoded audio."""
    expected_bytes = SAMPLES_PER_PKT * 2  # uint16 = 2 bytes each
    while not shutdown_event.is_set():
        try:
            data, _ = sock.recvfrom(expected_bytes * 2)
        except socket.timeout:
            continue
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
    global leftover, is_responding, listen_state

    output = np.zeros(frames, dtype=np.float32)
    write_pos = 0
    needed = frames

    with queue_lock:
        responding = is_responding

    if responding:
        # Drain TTS audio from response_queue
        if len(leftover) > 0:
            use = min(len(leftover), needed)
            output[write_pos : write_pos + use] = leftover[:use]
            leftover = leftover[use:]
            write_pos += use
            needed -= use

        while needed > 0:
            with queue_lock:
                if not response_queue:
                    break
                if response_queue[0] is None:
                    response_queue.popleft()  # discard this sentinel
                    # Only stop if there's nothing else queued
                    if not response_queue:
                        is_responding = False
                        leftover = np.zeros(0, dtype=np.float32)
                        with state_lock:
                            listen_state = ListenState.IDLE
                    # Either way, stop filling this callback frame
                    break
                chunk = response_queue.popleft()

            if len(chunk) <= needed:
                output[write_pos : write_pos + len(chunk)] = chunk
                write_pos += len(chunk)
                needed -= len(chunk)
            else:
                output[write_pos : write_pos + needed] = chunk[:needed]
                leftover = chunk[needed:]
                needed = 0
    else:
        # Normal mic passthrough — existing logic unchanged
        if len(leftover) > 0:
            use = min(len(leftover), needed)
            output[write_pos : write_pos + use] = leftover[:use]
            leftover = leftover[use:]
            write_pos += use
            needed -= use

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


def warmup_connections() -> None:
    """Fire a silent warmup request to the DeepSeek LLM on startup.

    DeepSeek caches the KV state of the system prompt after the first call.
    Subsequent calls with the same system prompt get a cache hit, which is
    the main source of first-interaction latency. This primes that cache
    before the user ever speaks. Runs in a daemon thread; never touches
    pipeline state, history, or ESP32 audio.
    """
    print(f"{ts()} [WARMUP] Priming DeepSeek prompt cache...", flush=True)

    llm_client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    try:
        t0 = time.monotonic()
        stream = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": "Hey Elio, how are you doing?"},
            ],
            extra_body={
                "provider": {"sort": "throughput"},
                "preferred_max_latency": {"p90": 2},
            },
            stream=True,
            max_tokens=60,
        )
        # Drain the stream fully so the system prompt KV cache is committed
        for _ in stream:
            pass
        print(
            f"{ts()} [WARMUP] Cache primed ({time.monotonic() - t0:.2f}s)", flush=True
        )
    except Exception as exc:
        print(f"{ts()} [WARMUP] Warmup failed (non-fatal): {exc}", flush=True)


def main() -> None:
    if RECORDING_MODE:
        print(
            f"{ts()} *** RECORDING MODE ACTIVE — all voice pipeline threads disabled ***"
        )
    else:
        print(
            f"{ts()} Using Groq STT model '{GROQ_MODEL}', OpenRouter TTS model '{TTS_MODEL}'"
        )
        load_silero_vad()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)
    print(f"{ts()} Listening for UDP audio on port {UDP_PORT}...")

    # Start all background threads
    threads = []
    for target, args in [
        (receive_loop, (sock,)),
    ]:
        t = threading.Thread(target=target, args=args, daemon=True)
        t.start()
        threads.append(t)

    if not RECORDING_MODE:
        for target, args in [
            (control_listener, ()),
            (vad_accumulator_loop, ()),
            (transcription_loop, ()),
            (llm_loop, ()),
            (tts_loop, ()),
            (audio_dispatch_loop, ()),
        ]:
            t = threading.Thread(target=target, args=args, daemon=True)
            t.start()
            threads.append(t)

        # Fire warmup in background — don't block startup or the pre-buffer wait
        warmup_thread = threading.Thread(
            target=warmup_connections, daemon=True, name="Warmup"
        )
        warmup_thread.start()

    print(f"{ts()} Waiting for {PREBUFFER_PKTS} packets to pre-buffer...")
    deadline = time.monotonic() + 10.0  # wait at most 10 seconds
    while True:
        with queue_lock:
            if len(packet_queue) >= PREBUFFER_PKTS:
                break
        if time.monotonic() > deadline:
            print(f"{ts()} WARNING: No audio from ESP32 after 10s — continuing anyway.")
            break
        time.sleep(0.01)

    print(f"{ts()} Starting playback. Press Ctrl+C to stop.")
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=SAMPLES_PER_PKT,
    ):
        try:
            while not shutdown_event.is_set():
                sd.sleep(200)
        except KeyboardInterrupt:
            print(f"\n{ts()} Shutting down...")

        shutdown_event.set()

        # Unblock any thread stuck on queue.get() with sentinel values
        llm_queue.put(None)
        tts_queue.put(None)
        transcribe_queue.put(None)

        for t in threads:
            t.join(timeout=3.0)

        print(f"{ts()} All threads stopped. Goodbye.")


if __name__ == "__main__":
    main()
