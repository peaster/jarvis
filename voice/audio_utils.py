"""Audio format conversion utilities.

Handles conversion between browser WebAudio formats and model requirements:
- Browser sends: Float32 or Int16 PCM @ 16kHz
- Whisper expects: Float32 @ 16kHz
- VibeVoice outputs: Float32 @ 24kHz
- Browser receives: Int16 PCM @ 24kHz
"""

import io
import struct
import numpy as np
from scipy import signal


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert Float32 PCM [-1.0, 1.0] to Int16 PCM [-32768, 32767].

    Args:
        audio: Float32 audio array normalized to [-1.0, 1.0]

    Returns:
        Int16 audio array
    """
    # Clip to valid range and scale
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert Int16 PCM [-32768, 32767] to Float32 PCM [-1.0, 1.0].

    Args:
        audio: Int16 audio array

    Returns:
        Float32 audio array normalized to [-1.0, 1.0]
    """
    return audio.astype(np.float32) / 32768.0


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample audio from source to destination sample rate.

    Args:
        audio: Input audio array
        src_rate: Source sample rate (e.g., 16000)
        dst_rate: Destination sample rate (e.g., 24000)

    Returns:
        Resampled audio array
    """
    if src_rate == dst_rate:
        return audio

    # Calculate resampling ratio
    num_samples = int(len(audio) * dst_rate / src_rate)

    # Use scipy's resample for high-quality resampling
    resampled = signal.resample(audio, num_samples)

    return resampled.astype(audio.dtype)


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to target dB level.

    Args:
        audio: Input audio array (Float32)
        target_db: Target RMS level in dB (default -20 dB)

    Returns:
        Normalized audio array
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio

    # Calculate target RMS from dB
    target_rms = 10 ** (target_db / 20)

    # Scale audio
    return audio * (target_rms / rms)


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV file bytes.

    Args:
        audio: Audio array (Float32 or Int16)
        sample_rate: Sample rate in Hz

    Returns:
        WAV file as bytes
    """
    # Convert to Int16 if Float32
    if audio.dtype == np.float32:
        audio = float32_to_int16(audio)

    # Ensure Int16
    audio = audio.astype(np.int16)

    # Create WAV header
    buffer = io.BytesIO()

    # WAV header parameters
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(audio) * block_align

    # Write RIFF header
    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 36 + data_size))  # File size - 8
    buffer.write(b'WAVE')

    # Write fmt chunk
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))  # Chunk size
    buffer.write(struct.pack('<H', 1))   # Audio format (PCM)
    buffer.write(struct.pack('<H', num_channels))
    buffer.write(struct.pack('<I', sample_rate))
    buffer.write(struct.pack('<I', byte_rate))
    buffer.write(struct.pack('<H', block_align))
    buffer.write(struct.pack('<H', bits_per_sample))

    # Write data chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<I', data_size))
    buffer.write(audio.tobytes())

    return buffer.getvalue()


def bytes_to_int16_audio(data: bytes) -> np.ndarray:
    """Convert raw bytes to Int16 audio array.

    Args:
        data: Raw PCM bytes (Int16 little-endian)

    Returns:
        Int16 numpy array
    """
    return np.frombuffer(data, dtype=np.int16)


def bytes_to_float32_audio(data: bytes) -> np.ndarray:
    """Convert raw bytes to Float32 audio array.

    Assumes input is Int16 PCM bytes from browser.

    Args:
        data: Raw PCM bytes (Int16 little-endian)

    Returns:
        Float32 numpy array normalized to [-1.0, 1.0]
    """
    int16_audio = np.frombuffer(data, dtype=np.int16)
    return int16_to_float32(int16_audio)


def chunk_audio(audio: np.ndarray, chunk_samples: int) -> list[np.ndarray]:
    """Split audio into fixed-size chunks.

    Args:
        audio: Input audio array
        chunk_samples: Number of samples per chunk

    Returns:
        List of audio chunks
    """
    chunks = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) > 0:
            chunks.append(chunk)
    return chunks


def calculate_audio_duration(num_samples: int, sample_rate: int) -> float:
    """Calculate duration in seconds from sample count.

    Args:
        num_samples: Number of audio samples
        sample_rate: Sample rate in Hz

    Returns:
        Duration in seconds
    """
    return num_samples / sample_rate
