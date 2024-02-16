from typing import TypedDict, cast

import requests
import torch
from transformers import AutomaticSpeechRecognitionPipeline, pipeline


# Exception for failing to process audio
class AudioProcessingException(Exception):
    pass


# define a base abstract class for all listeners
class BaseListener:
    def __call__(self, audio: bytes) -> bool | str:
        raise NotImplementedError


class SupabaseFunctionListener(BaseListener):
    def __init__(self, supabase_url, supabase_key):
        super().__init__()
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key

    def __call__(self, audio: bytes) -> bool:
        files = {"file": audio}
        response = requests.post(
            url=self.supabase_url,
            files=files,
            headers={
                "apikey": self.supabase_key,
                "Content-Type": "audio/wav",
            },
        )

        # Check the response status code
        if response.status_code == 200:
            return True
        else:
            raise AudioProcessingException(
                f"Failed to send audio chunk, received status {response.status_code}"
            )


class TranscribeResult(TypedDict):
    text: str


class LocalPipelineLisenter(BaseListener):
    pipeline: AutomaticSpeechRecognitionPipeline

    def __init__(
        self,
        model_name: str = "BlueRaccoon/whisper-small-en",
        token: str | None = None,
    ):
        super().__init__()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # pipe = pipeline("automatic-speech-recognition", model="BlueRaccoon/whisper-small-en")
        self.pipeline = cast(
            AutomaticSpeechRecognitionPipeline,
            pipeline(
                task="automatic-speech-recognition",
                framework="pt",
                model=model_name,
                device=device,
                token=token,
            ),
        )

    def __call__(self, audio: bytes) -> str:
        """
        Transcribe raw audio to text
        """
        try:
            result = cast(TranscribeResult, self.pipeline(audio))
            return result["text"]
        except Exception:
            raise AudioProcessingException("Failed to process audio")
