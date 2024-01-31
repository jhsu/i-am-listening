import os
import time
from typing import Callable, Optional

import numpy as np
import openai
import pyaudio  # noqa: F401
import speech_recognition as sr
import torch
from dotenv import load_dotenv
from notetaker import NoteTaker
from pyannote.audio import Pipeline
from whisper import Whisper, load_model

load_dotenv()

pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token=os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
)


def process_segment(
    audio_segment: sr.AudioData,
    audio_model: Whisper,
    sample_rate: int = 16000,
    callback: Optional[Callable] = None,
) -> None:
    audio_data = np.frombuffer(audio_segment.get_wav_data(), dtype=np.int16)

    # Convert in-ram buffer to something the model can use directly without needing a temp file.
    # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
    # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    # Ensure the audio is in the shape (channel, time), assuming mono audio
    audio_data = np.expand_dims(audio_data, axis=0)

    # Convert the numpy array to a torch tensor
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

    # Use the pipeline to detect speech activity
    speech_activity = pipeline({"waveform": audio_tensor, "sample_rate": sample_rate})

    has_speech = any(
        label == "SPEECH"
        for segment, _, label in speech_activity.itertracks(yield_label=True)
    )

    if has_speech:
        print("Processing audio segment...")
        result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
        if isinstance(result["text"], str):
            text = result["text"].strip()
            if callback and text != "":
                callback(text)
            print(f"received text: '{text}'")


def main() -> None:
    recognizer: sr.Recognizer = sr.Recognizer()
    sample_rate = 16000
    mic: sr.Microphone = sr.Microphone(sample_rate=sample_rate)

    notetaker = NoteTaker(
        db_path=os.path.join(os.getcwd(), "notes.db"), debounce_time=1
    )

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        recognizer.dynamic_energy_threshold = True

    model = "tiny.en"

    print("Listening...")

    audio_model = load_model(model)

    try:
        while True:
            with mic as source:
                try:
                    # Adjust the duration parameter to control how long it listens before processing
                    audio: sr.AudioData = recognizer.listen(
                        source, timeout=5.0, phrase_time_limit=5.0
                    )
                    process_segment_in_chunks(
                        audio,
                        sample_rate=sample_rate,
                        audio_model=audio_model,
                        callback=notetaker.append,
                    )
                except sr.WaitTimeoutError:
                    pass  # Timeout reached, no speech detected

                time.sleep(0.5)  # Sleep for a short time before listening again
    except KeyboardInterrupt:
        notetaker.flush()
        print("exiting.")


if __name__ == "__main__":
    main()
def process_segment_in_chunks(
    audio_segment: sr.AudioData,
    audio_model: Whisper,
    chunk_size: int = 4000,
    sample_rate: int = 16000,
    callback: Optional[Callable] = None,
    use_openai: bool = False,
) -> None:
    audio_data = np.frombuffer(audio_segment.get_wav_data(), dtype=np.int16)

    # Convert in-ram buffer to something the model can use directly without needing a temp file.
    # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
    # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    # Ensure the audio is in the shape (channel, time), assuming mono audio
    audio_data = np.expand_dims(audio_data, axis=0)

    # Convert the numpy array to a torch tensor
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

    # Use the pipeline to detect speech activity
    speech_activity = pipeline({"waveform": audio_tensor, "sample_rate": sample_rate})

    has_speech = any(
        label == "SPEECH"
        for segment, _, label in speech_activity.itertracks(yield_label=True)
    )

    if has_speech:
        print("Processing audio segment in chunks...")
        for i in range(0, len(audio_np), chunk_size):
            chunk = audio_np[i:i+chunk_size]
            if use_openai:
                result = openai.client.audio.transcriptions.create(
                    audio=chunk,
                    model="en-US",
                    token=os.environ.get("OPENAI_API_KEY"),
                )
            else:
                result = audio_model.transcribe(chunk, fp16=torch.cuda.is_available())
            if isinstance(result["text"], str):
                text = result["text"].strip()
                if callback and text != "":
                    callback(text)
                print(f"received text: '{text}'")
