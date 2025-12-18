"""Automatic Speech Recognition (ASR) engine using Whisper.

Provides speech-to-text transcription with support for:
- Local GPU inference via transformers
- Async transcription for non-blocking operation
- Automatic audio preprocessing (resampling, normalization)
"""

import asyncio
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from voice.audio_utils import int16_to_float32, resample_audio


class ASREngine:
    """Whisper-based ASR engine for speech-to-text transcription.

    Uses the transformers library to run Whisper models locally on GPU.
    Optimized for low-latency transcription with FP16 inference.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3-turbo",
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        language: str = "en",
    ):
        """Initialize the ASR engine.

        Args:
            model_id: HuggingFace model identifier for Whisper
            device: Device to run inference on ("cuda" or "cpu")
            torch_dtype: PyTorch dtype for model (auto-detected if None)
            language: Language code for transcription (default "en")
        """
        self.device = device
        self.language = language

        # Auto-detect dtype based on device
        if torch_dtype is None:
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self.torch_dtype = torch_dtype

        print(f"Loading ASR model: {model_id} on {device} ({torch_dtype})")

        # Load model and processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        # Create pipeline for easy inference
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        # Target sample rate for Whisper
        self.sample_rate = 16000

        print(f"ASR engine ready (sample rate: {self.sample_rate}Hz)")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        return_timestamps: bool = False,
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio array (Float32 or Int16)
            sample_rate: Sample rate of input audio
            return_timestamps: Whether to return word timestamps

        Returns:
            Transcribed text string
        """
        # Convert Int16 to Float32 if needed
        if audio.dtype == np.int16:
            audio = int16_to_float32(audio)

        # Ensure Float32
        audio = audio.astype(np.float32)

        # Resample to 16kHz if needed
        if sample_rate != self.sample_rate:
            audio = resample_audio(audio, sample_rate, self.sample_rate)

        # Skip very short audio (< 0.1 seconds)
        if len(audio) < self.sample_rate * 0.1:
            return ""

        # Run transcription
        generate_kwargs = {
            "language": self.language,
            "task": "transcribe",
        }

        result = self.pipe(
            {"sampling_rate": self.sample_rate, "raw": audio},
            return_timestamps=return_timestamps,
            generate_kwargs=generate_kwargs,
        )

        # Extract text
        text = result.get("text", "").strip()

        return text

    async def transcribe_async(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        return_timestamps: bool = False,
    ) -> str:
        """Async wrapper for transcription.

        Runs transcription in a thread pool to avoid blocking the event loop.

        Args:
            audio: Audio array (Float32 or Int16)
            sample_rate: Sample rate of input audio
            return_timestamps: Whether to return word timestamps

        Returns:
            Transcribed text string
        """
        return await asyncio.to_thread(
            self.transcribe, audio, sample_rate, return_timestamps
        )

    def transcribe_chunks(
        self,
        audio_chunks: list[np.ndarray],
        sample_rate: int = 16000,
    ) -> list[str]:
        """Transcribe multiple audio chunks.

        Useful for processing segmented audio from VAD.

        Args:
            audio_chunks: List of audio arrays
            sample_rate: Sample rate of input audio

        Returns:
            List of transcribed text strings
        """
        transcripts = []
        for chunk in audio_chunks:
            text = self.transcribe(chunk, sample_rate)
            if text:
                transcripts.append(text)
        return transcripts

    async def transcribe_chunks_async(
        self,
        audio_chunks: list[np.ndarray],
        sample_rate: int = 16000,
    ) -> list[str]:
        """Async version of transcribe_chunks."""
        return await asyncio.to_thread(
            self.transcribe_chunks, audio_chunks, sample_rate
        )

    def warmup(self):
        """Warm up the model with a dummy inference.

        Call this during startup to ensure first real inference is fast.
        """
        dummy_audio = np.zeros(self.sample_rate, dtype=np.float32)  # 1 second of silence
        _ = self.transcribe(dummy_audio)
        print("ASR engine warmed up")

    @property
    def model_sample_rate(self) -> int:
        """Get the expected input sample rate."""
        return self.sample_rate
