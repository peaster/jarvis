# Project Guide

## Voice / TTS Configuration

This project uses **VibeVoice-Realtime-0.5B** for text-to-speech synthesis.

### TTS Modes

The `tts_mode` setting in voice configuration controls how text is sent to VibeVoice:

| Mode | Latency | Quality | Use Case |
|------|---------|---------|----------|
| `"full"` (default) | Higher (waits for LLM to complete) | Best - natural prosody | Long-form content |
| `"streaming"` | Lower (~300ms) | Lower - chunk breaks | Interactive chat |

**Why this matters**: VibeVoice-Realtime supports up to 8K tokens (~10 minutes of audio) in a single generation. The `"full"` mode sends the complete LLM response as one call, allowing VibeVoice to handle internal windowing for optimal prosody and continuity. The `"streaming"` mode chunks text into 100-500 character pieces, which can cause awkward breaks at chunk boundaries.

### TTS Speed

The `tts_speed` setting controls the playback speed of synthesized speech:

| Speed | Effect |
|-------|--------|
| `1.0` | Normal speed (default) |
| `1.2` | 20% faster - recommended for conversational use |
| `1.5` | 50% faster |
| `0.8` | 20% slower |

Uses librosa time-stretching to change speed without affecting pitch.

### Configuration

In `mcp_agent_config.json`:
```json
{
  "voice": {
    "enabled": true,
    "tts_mode": "full",
    "tts_speed": 1.2,
    "speaker": "Carter"
  }
}
```

### Key Files

- `voice/tts.py` - TTSEngine class with `synthesize_full_text()` and `synthesize_streaming()` methods
- `voice/audio_utils.py` - Audio processing utilities including `time_stretch_audio()` for speed control
- `web.py` - VoiceConfig dataclass and `process_agent_response()` which handles TTS mode switching
- `VibeVoice/` - Submodule containing the VibeVoice library

### References

- [VibeVoice Project Page](https://microsoft.github.io/VibeVoice/)
- [VibeVoice-Realtime-0.5B on HuggingFace](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
- [VibeVoice GitHub Repository](https://github.com/microsoft/VibeVoice)
- [VibeVoice-Realtime Documentation](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md)
- [Technical Report (arXiv)](https://arxiv.org/abs/2508.19205)
