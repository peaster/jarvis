"""Text-to-Speech (TTS) engine using VibeVoice.

Provides streaming text-to-speech synthesis with:
- Real-time audio generation as text tokens arrive
- Low latency (~300ms first audible)
- Support for different speakers/voices
"""

import asyncio
import copy
import glob
import os
import threading
from pathlib import Path
from typing import AsyncGenerator, Callable, Iterator, Optional

import numpy as np
import torch

from voice.audio_utils import float32_to_int16, time_stretch_audio


class DictToAttr:
    """Wrapper that allows dict-like objects to be accessed as attributes.

    This is needed because VibeVoice voice cache files contain serialized
    model outputs that become dicts when loaded, but the model code expects
    objects with attribute access.
    """

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict) and not isinstance(value, DictToAttr):
                value = DictToAttr(value)
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def __iter__(self):
        return iter(self.__dict__.keys())

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def get(self, key, default=None):
        return getattr(self, key, default)


def find_voice_presets() -> dict[str, str]:
    """Find available voice preset files.

    Searches for .pt files in known VibeVoice voice directories.
    The VOICE_PRESETS_PATH environment variable takes priority if set.

    Returns:
        Dictionary mapping voice names (lowercase) to file paths.
    """
    voice_presets = {}

    # Build search paths, environment variable takes priority
    search_paths = []

    custom_path = os.getenv("VOICE_PRESETS_PATH")
    if custom_path:
        search_paths.append(Path(custom_path))

    search_paths.extend([
        # Docker baked-in path
        Path("/opt/VibeVoice/demo/voices/streaming_model"),
        # Relative to this file (local development)
        Path(__file__).parent.parent / "VibeVoice" / "demo" / "voices" / "streaming_model",
        # Common installation paths
        Path.home() / "VibeVoice" / "demo" / "voices" / "streaming_model",
    ])

    for search_path in search_paths:
        if search_path.exists():
            for pt_file in search_path.glob("*.pt"):
                # Extract name: "en-Carter_man.pt" -> "carter"
                name = pt_file.stem.lower()
                # Also create shorter alias without language prefix
                if "-" in name:
                    short_name = name.split("-", 1)[1].split("_")[0]
                    voice_presets[short_name] = str(pt_file)
                voice_presets[name] = str(pt_file)

    return voice_presets


class TTSEngine:
    """VibeVoice-based TTS engine for streaming text-to-speech.

    Generates audio in real-time as text is received, enabling
    low-latency voice responses during LLM streaming.

    Note: Requires VibeVoice installation:
        git clone https://github.com/microsoft/VibeVoice.git
        cd VibeVoice && pip install -e .
    """

    def __init__(
        self,
        model_id: str = "microsoft/VibeVoice-Realtime-0.5B",
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        speaker_name: str = "Carter",
        speed: float = 1.0,
    ):
        """Initialize the TTS engine.

        Args:
            model_id: HuggingFace model identifier for VibeVoice
            device: Device to run inference on ("cuda" or "cpu")
            torch_dtype: PyTorch dtype for model (auto-detected if None)
            speaker_name: Voice/speaker to use for synthesis
            speed: Speech speed multiplier (1.2 = 20% faster)
        """
        self.device = device
        self.speaker_name = speaker_name
        self.speed = speed
        self.sample_rate = 24000  # VibeVoice output sample rate

        # Auto-detect dtype based on device
        # VibeVoice works best with bfloat16 on CUDA
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if "cuda" in device else torch.float32
        self.torch_dtype = torch_dtype

        print(f"Loading TTS model: {model_id} on {device} ({torch_dtype})")

        # Try to import VibeVoice
        try:
            from vibevoice import (
                VibeVoiceStreamingForConditionalGenerationInference,
                VibeVoiceStreamingProcessor,
            )

            self.processor = VibeVoiceStreamingProcessor.from_pretrained(model_id)
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
            ).to(device)
            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=5)
            self.tokenizer = self.processor.tokenizer
            self._use_vibevoice = True

            # Load voice presets
            self.voice_presets = find_voice_presets()
            if self.voice_presets:
                print(f"Found {len(self.voice_presets)} voice presets")
            else:
                print("WARNING: No voice presets found. TTS may not work correctly.")

            # Load the speaker's voice preset
            self.voice_cache = None
            self._load_voice_preset(speaker_name)
            print(f"TTS engine ready with VibeVoice (speaker: {speaker_name})")

        except ImportError:
            print("WARNING: VibeVoice not installed. Using fallback TTS.")
            print("Install with: git clone https://github.com/microsoft/VibeVoice.git && cd VibeVoice && pip install -e .")
            self._use_vibevoice = False
            self._setup_fallback()

    def _setup_fallback(self):
        """Set up a fallback TTS engine (e.g., pyttsx3 or silent)."""
        # For now, we'll use a simple sine wave generator as placeholder
        # In production, you might want to use pyttsx3, edge-tts, or similar
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.voice_presets = {}
        self.voice_cache = None

    def _load_voice_preset(self, speaker_name: str):
        """Load voice preset for a speaker.

        Args:
            speaker_name: Name of the speaker to load
        """
        if not self._use_vibevoice or not self.voice_presets:
            return

        # Find matching voice preset
        speaker_lower = speaker_name.lower()
        voice_path = None

        # Try exact match first
        if speaker_lower in self.voice_presets:
            voice_path = self.voice_presets[speaker_lower]
        else:
            # Try partial match
            for name, path in self.voice_presets.items():
                if speaker_lower in name or name in speaker_lower:
                    voice_path = path
                    break

        if voice_path is None:
            # Use first available voice as fallback
            voice_path = list(self.voice_presets.values())[0]
            print(f"WARNING: No voice preset found for '{speaker_name}', using: {voice_path}")

        # Load the voice cache
        print(f"Loading voice preset: {voice_path}")
        raw_cache = torch.load(voice_path, map_location=self.device, weights_only=False)

        # Move tensors to correct device/dtype, but preserve DynamicCache objects
        def move_to_device(obj, dtype, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(dtype=dtype, device=device)
            elif isinstance(obj, dict):
                return {k: move_to_device(v, dtype, device) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(move_to_device(v, dtype, device) for v in obj)
            elif hasattr(obj, "to"):
                # Handle objects like DynamicCache that have a .to() method
                return obj.to(device)
            return obj

        raw_cache = move_to_device(raw_cache, self.torch_dtype, self.device)

        # Wrap the model output dicts in DictToAttr so getattr() works on them
        # The voice cache structure is: {"lm": {...}, "tts_lm": {...}, "neg_lm": {...}, "neg_tts_lm": {...}}
        # Each inner dict needs attribute access for the VibeVoice model
        self.voice_cache = {}
        for key, value in raw_cache.items():
            if isinstance(value, dict):
                self.voice_cache[key] = DictToAttr(value)
            else:
                self.voice_cache[key] = value

        self.speaker_name = speaker_name

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize

        Returns:
            Float32 audio array at 24kHz
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32)

        if self._use_vibevoice:
            return self._synthesize_vibevoice(text)
        else:
            return self._synthesize_fallback(text)

    def _synthesize_vibevoice(self, text: str) -> np.ndarray:
        """Synthesize using VibeVoice model."""
        if self.voice_cache is None:
            print("WARNING: No voice cache loaded, using fallback")
            return self._synthesize_fallback(text)

        # Clean text
        text = text.replace("'", "'").replace('"', '"').replace('"', '"')

        # Prepare inputs using the processor with cached voice prompt
        inputs = self.processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=self.voice_cache,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move tensors to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=1.5,
                tokenizer=self.processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(self.voice_cache),
            )

        # Extract audio from outputs
        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio = outputs.speech_outputs[0]
            if torch.is_tensor(audio):
                # Convert to float32 before numpy (numpy doesn't support bfloat16)
                audio = audio.float().cpu().numpy()
            audio_np = audio.squeeze().astype(np.float32)
            return audio_np
        else:
            print("WARNING: No audio output from VibeVoice")
            return np.array([], dtype=np.float32)

    def _synthesize_fallback(self, text: str) -> np.ndarray:
        """Fallback synthesis when VibeVoice not available.

        Generates a simple tone as placeholder.
        """
        # Generate a short beep as placeholder
        duration = min(0.5, len(text) * 0.02)  # Roughly estimate duration
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        return audio.astype(np.float32)

    def stream(
        self,
        text: str,
        cfg_scale: float = 1.5,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        """Stream audio generation using VibeVoice's native streaming.

        Uses AudioStreamer to yield audio chunks as they're generated,
        producing smooth continuous audio without gaps.

        Args:
            text: Text to synthesize
            cfg_scale: Classifier-free guidance scale (default 1.5)
            stop_event: Optional event to signal early stopping

        Yields:
            Float32 audio chunks as they're generated
        """
        if not text or not text.strip():
            return

        if not self._use_vibevoice:
            # Fallback: yield single chunk
            yield self._synthesize_fallback(text)
            return

        if self.voice_cache is None:
            print("WARNING: No voice cache loaded, using fallback")
            yield self._synthesize_fallback(text)
            return

        # Import AudioStreamer from VibeVoice
        from vibevoice.modular.streamer import AudioStreamer

        # Clean text
        text = text.replace("'", "'").replace('"', '"').replace('"', '"')

        # Prepare inputs using the processor with cached voice prompt
        inputs = self.processor.process_input_with_cached_prompt(
            text=text.strip(),
            cached_prompt=self.voice_cache,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move tensors to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        # Create audio streamer and stop signal
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        stop_signal = stop_event or threading.Event()

        # Run generation in background thread
        thread = threading.Thread(
            target=self._run_streaming_generation,
            kwargs={
                "inputs": inputs,
                "audio_streamer": audio_streamer,
                "cfg_scale": cfg_scale,
                "stop_event": stop_signal,
            },
            daemon=True,
        )
        thread.start()

        # Yield audio chunks as they arrive from the queue
        try:
            for audio_chunk in audio_streamer.get_stream(0):
                if torch.is_tensor(audio_chunk):
                    audio_chunk = audio_chunk.float().cpu().numpy()
                chunk = audio_chunk.squeeze().astype(np.float32)
                if chunk.size > 0:
                    yield chunk
        finally:
            stop_signal.set()
            audio_streamer.end()
            thread.join(timeout=5.0)

    def _run_streaming_generation(
        self,
        inputs: dict,
        audio_streamer,
        cfg_scale: float,
        stop_event: threading.Event,
    ) -> None:
        """Run VibeVoice generation in background thread with audio streaming.

        Args:
            inputs: Prepared model inputs
            audio_streamer: AudioStreamer to receive audio chunks
            cfg_scale: Classifier-free guidance scale
            stop_event: Event to check for early stopping
        """
        try:
            with torch.no_grad():
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop_event.is_set,
                    verbose=False,
                    all_prefilled_outputs=copy.deepcopy(self.voice_cache),
                )
        except Exception as e:
            print(f"TTS streaming generation error: {e}")
        finally:
            audio_streamer.end()

    async def synthesize_async(self, text: str) -> np.ndarray:
        """Async wrapper for synthesis."""
        return await asyncio.to_thread(self.synthesize, text)

    async def synthesize_full_text(
        self,
        text: str,
        on_audio_chunk: Callable[[bytes, bool], None],
    ):
        """Synthesize complete text using VibeVoice's native streaming.

        Sends full text to VibeVoice in a single call, letting it handle
        internal windowing for optimal prosody and audio quality. This
        produces smoother, more natural speech than chunked synthesis.

        Args:
            text: Complete text to synthesize
            on_audio_chunk: Async callback receiving (audio_bytes, is_final)
        """
        if not text or not text.strip():
            await on_audio_chunk(b"", True)
            return

        try:
            # Use native streaming - yields chunks as generated
            for audio_chunk in self.stream(text):
                audio_bytes = self._audio_to_bytes(audio_chunk)
                await on_audio_chunk(audio_bytes, False)
        except Exception as e:
            print(f"TTS full text error: {e}")

        # Signal completion
        await on_audio_chunk(b"", True)

    async def synthesize_streaming(
        self,
        text_generator: AsyncGenerator[str, None],
        on_audio_chunk: Callable[[bytes, bool], None],
        min_chars: int = 100,
        max_chars: int = 500,
    ):
        """Stream text tokens to audio chunks using native VibeVoice streaming.

        Accumulates text until we have a complete sentence, then uses
        VibeVoice's AudioStreamer for smooth continuous audio generation.

        Args:
            text_generator: Async generator yielding text chunks
            on_audio_chunk: Async callback receiving (audio_bytes, is_final)
            min_chars: Minimum characters before generating audio (default 100)
            max_chars: Maximum characters before forcing synthesis (default 500)
        """
        text_buffer = ""

        # Characters that indicate a good break point for synthesis
        break_chars = ".!?\n"

        async for text_chunk in text_generator:
            if not text_chunk:
                continue

            text_buffer += text_chunk

            # Find last break point in buffer
            break_pos = -1
            for i in range(len(text_buffer) - 1, -1, -1):
                if text_buffer[i] in break_chars:
                    break_pos = i + 1
                    break

            should_synthesize = False
            synth_text = ""

            # Synthesize at natural break points if we have enough text
            if break_pos > 0 and break_pos >= min_chars:
                synth_text = text_buffer[:break_pos]
                text_buffer = text_buffer[break_pos:].lstrip()
                should_synthesize = True
            # Force synthesis if buffer is getting too large (prevents runaway)
            elif len(text_buffer) >= max_chars:
                synth_text = text_buffer
                text_buffer = ""
                should_synthesize = True

            if should_synthesize and synth_text.strip():
                # Use native streaming - yields chunks as generated
                try:
                    for audio_chunk in self.stream(synth_text):
                        audio_bytes = self._audio_to_bytes(audio_chunk)
                        await on_audio_chunk(audio_bytes, False)
                except Exception as e:
                    print(f"TTS streaming error: {e}")

        # Final chunk - synthesize any remaining text
        if text_buffer.strip():
            try:
                for audio_chunk in self.stream(text_buffer):
                    audio_bytes = self._audio_to_bytes(audio_chunk)
                    await on_audio_chunk(audio_bytes, False)
            except Exception as e:
                print(f"TTS streaming error: {e}")

        # Signal completion
        await on_audio_chunk(b"", True)

    def _audio_to_bytes(self, audio: np.ndarray) -> bytes:
        """Convert audio array to Int16 PCM bytes for streaming.

        Applies time stretching if speed != 1.0.

        Args:
            audio: Float32 audio array

        Returns:
            Int16 PCM bytes (little-endian)
        """
        if self.speed != 1.0:
            audio = time_stretch_audio(audio, self.speed)
        audio_int16 = float32_to_int16(audio)
        return audio_int16.tobytes()

    def synthesize_to_bytes(self, text: str) -> bytes:
        """Synthesize text and return as Int16 PCM bytes.

        Args:
            text: Text to synthesize

        Returns:
            Int16 PCM bytes at 24kHz
        """
        audio = self.synthesize(text)
        return self._audio_to_bytes(audio)

    async def synthesize_to_bytes_async(self, text: str) -> bytes:
        """Async version of synthesize_to_bytes."""
        return await asyncio.to_thread(self.synthesize_to_bytes, text)

    def warmup(self):
        """Warm up the model with a dummy inference."""
        _ = self.synthesize("Hello.")
        print("TTS engine warmed up")

    @property
    def output_sample_rate(self) -> int:
        """Get the output audio sample rate."""
        return self.sample_rate

    def set_speaker(self, speaker_name: str):
        """Change the speaker/voice.

        Args:
            speaker_name: Name of the speaker to use
        """
        if speaker_name.lower() != self.speaker_name.lower():
            self._load_voice_preset(speaker_name)
