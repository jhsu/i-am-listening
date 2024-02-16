import os
from typing import Callable, Optional

import numpy as np

# import openai
import pyaudio  # noqa: F401
import requests
import speech_recognition as sr
import torch
from dotenv import load_dotenv
from listeners import BaseListener, LocalPipelineLisenter
from notetaker import NoteTaker
from pyannote.audio import Pipeline
from whisper import Whisper

load_dotenv()

# Used to detect voice activity
activity_pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token=os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
)


# def process_segment(
#     audio_segment: sr.AudioData,
#     audio_model: Whisper,
#     sample_rate: int = 16000,
#     callback: Optional[Callable] = None,
# ) -> None:
#     audio_data = np.frombuffer(audio_segment.get_wav_data(), dtype=np.int16)

#     # Convert in-ram buffer to something the model can use directly without needing a temp file.
#     # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
#     # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
#     audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

#     # Ensure the audio is in the shape (channel, time), assuming mono audio
#     audio_data = np.expand_dims(audio_data, axis=0)

#     # Convert the numpy array to a torch tensor
#     audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

#     # Use the pipeline to detect speech activity
#     speech_activity = activity_pipeline(
#         {"waveform": audio_tensor, "sample_rate": sample_rate}
#     )

#     has_speech = any(
#         label == "SPEECH"
#         for segment, _, label in speech_activity.itertracks(yield_label=True)
#     )

#     if has_speech:
#         print("Processing audio segment...")
#         result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
#         if isinstance(result["text"], str):
#             text = result["text"].strip()
#             if callback and text != "":
#                 callback(text)
#             print(f"received text: '{text}'")


def process_segment_in_chunks(
    audio_segment: sr.AudioData,
    transcribe: BaseListener,
    # chunk_size: int = 4000,
    sample_rate: int = 16000,
    callback: Optional[Callable[[str], None]] = None,
) -> None:
    audio_data = np.frombuffer(audio_segment.get_wav_data(), dtype=np.int16)

    # Ensure the audio is in the shape (channel, time), assuming mono audio
    audio_data = np.expand_dims(audio_data, axis=0)

    # Convert the numpy array to a torch tensor
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

    # Use the pipeline to detect speech activity
    speech_activity = activity_pipeline(
        {"waveform": audio_tensor, "sample_rate": sample_rate}
    )

    has_speech = any(
        label == "SPEECH"
        for segment, _, label in speech_activity.itertracks(yield_label=True)
    )

    if has_speech:
        # Transcribe audio using sm english
        # result = pipelines["sm_en"](audio_segment.get_wav_data())["text"]
        result = transcribe(audio_segment.get_wav_data())
        if callback and isinstance(result, str):
            callback(result)
        print(f"result: {result}")


def main() -> None:
    recognizer = sr.Recognizer()
    sample_rate = 16000
    mic: sr.Microphone = sr.Microphone(sample_rate=sample_rate)

    notetaker = NoteTaker(
        db_path=os.path.join(os.getcwd(), "notes.db"), debounce_time=1
    )

    transcribe = LocalPipelineLisenter()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        recognizer.dynamic_energy_threshold = True

    print("Listening...")

    try:
        while True:
            with mic as source:
                try:
                    # Adjust the duration parameter to control how long it listens before processing
                    audio = recognizer.listen(
                        source, timeout=None, phrase_time_limit=15.0
                    )
                    process_segment_in_chunks(
                        audio,
                        sample_rate=sample_rate,
                        callback=notetaker.append,
                        transcribe=transcribe,
                    )
                except sr.WaitTimeoutError:
                    pass  # Timeout reached, no speech detected
    except KeyboardInterrupt:
        print("exiting.")
    finally:
        notetaker.flush()
        # close the mic


if __name__ == "__main__":
    main()
