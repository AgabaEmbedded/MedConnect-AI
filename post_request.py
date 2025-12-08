import requests
import json
import base64
import pyaudio
import webrtcvad
import wave
import time

API_URL = "http://127.0.0.1:8000/conversation"
#API_URL = "https://medconnect-ai-4fnj.onrender.com/conversation"

# --- Audio Settings ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION = 30  # ms (10/20/30 allowed)
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)
SILENCE_LIMIT = 1.0  # seconds


vad = webrtcvad.Vad(2)
pa = pyaudio.PyAudio()

stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAME_SIZE
)


def record_audio():
    print("\n🎤 Recording... Speak now.")

    frames = []
    silence_counter = 0
    max_silence_frames = int(SILENCE_LIMIT * 1000 / FRAME_DURATION)

    try:
        while True:
            data = stream.read(FRAME_SIZE, exception_on_overflow=False)
            is_speech = vad.is_speech(data, RATE)

            if is_speech:
                frames.append(data)
                silence_counter = 0
            else:
                silence_counter += 1
                if frames:
                    frames.append(data)  # keep small trailing silence

            if silence_counter > max_silence_frames and frames:
                break

    except KeyboardInterrupt:
        print("Stopped manually.")

    print("📁 Saving audio to output.wav")

    # Save WAV properly
    output_file = "audio_file.wav"
    wf = wave.open(output_file, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    print("✔ Saved:", output_file)

    # Encode to base64
    with open(output_file, "rb") as file:
        audio_b64 = base64.b64encode(file.read()).decode("utf-8")

    return audio_b64


# ---- Conversation Loop ----

language = input("Enter language [english, hausa, igbo, yoruba]: ").strip()

while True:
    choice = input("\nUse Audio or Text? (a/t): ").strip().lower()

    if choice == "t":
        user_text = input("\n👤 YOU: ")
        payload = {
            "message": user_text,
            "language": language,
            "audio": "",
            "premium": True
        }

    elif choice == "a":
        audio_b64 = record_audio()
        payload = {
            "message": "",
            "language": language,
            "audio": audio_b64,
            "premium": False
        }

    else:
        print("❌ Invalid choice")
        continue

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()

        data = response.json()

        # Handle audio response
        if data.get("audio"):
            with open("response.wav", "wb") as f:
                f.write(base64.b64decode(data["audio"]))
            print("🎧 Assistant audio saved as response.wav")

        # Handle message + doctor id safely
        msg = data.get("message", "")
        doctor = data.get("doctorid", "")

        print(f"\n🤖 ASSISTANT: {msg}{doctor}")

    except Exception as e:
        print("❌ Error:", e)
