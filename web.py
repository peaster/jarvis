"""
Voice-Enabled Web Interface for MCP Agent

Full-duplex voice agent with:
- Speech-to-text via Whisper
- Text-to-speech via VibeVoice with streaming
- Real-time WebSocket communication
- Browser-side VAD for speech detection

Usage:
    python web.py
    python web.py --config custom_config.json --port 8000
"""

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from mcp_agent import MCPAgent, Config


@dataclass
class VoiceConfig:
    """Voice processing configuration."""
    enabled: bool = True
    asr_device: str = "cuda:0"
    tts_device: str = "cuda:1"
    asr_model: str = "openai/whisper-large-v3-turbo"
    tts_model: str = "microsoft/VibeVoice-Realtime-0.5B"
    speaker: str = "Carter"
    sample_rate_in: int = 16000
    sample_rate_out: int = 24000


@dataclass
class VoiceSession:
    """Manages state for a voice conversation session."""
    session_id: str
    history: Optional[list[dict]] = None
    audio_buffer: bytearray = field(default_factory=bytearray)
    is_speaking: bool = False


# Global state
agent: Optional[MCPAgent] = None
asr_engine = None
tts_engine = None
voice_config: Optional[VoiceConfig] = None
sessions: dict[str, VoiceSession] = {}


HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Voice MCP Agent</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: system-ui, -apple-system, sans-serif;
            background: #1a1a2e; color: #eee;
            height: 100vh; display: flex; flex-direction: column;
        }
        #header {
            padding: 16px 20px; background: #16213e;
            border-bottom: 1px solid #2d3748;
            display: flex; justify-content: space-between; align-items: center;
        }
        #header h1 { font-size: 1.1em; font-weight: 500; }
        .status { font-size: 0.8em; padding: 4px 8px; border-radius: 4px; }
        .connected { background: #065f46; color: #10b981; }
        .disconnected { background: #7f1d1d; color: #ef4444; }
        .listening { background: #78350f; color: #f59e0b; }
        .processing { background: #1e3a8a; color: #60a5fa; }
        .speaking { background: #5b21b6; color: #a78bfa; }

        #chat {
            flex: 1; overflow-y: auto; padding: 20px;
            display: flex; flex-direction: column; gap: 12px;
        }
        .msg {
            max-width: 85%; padding: 12px 16px; border-radius: 12px;
            line-height: 1.5; word-wrap: break-word;
        }
        .user { background: #4a5568; align-self: flex-end; }
        .assistant { background: #2d3748; align-self: flex-start; }
        .transcript { background: #374151; font-style: italic; opacity: 0.8; }
        .tool {
            background: #1e3a5f; font-size: 0.85em;
            border-left: 3px solid #3b82f6; margin-left: 20px; max-width: 75%;
        }
        .tool-name { color: #60a5fa; font-weight: 600; }
        .tool-result { color: #9ca3af; margin-top: 4px; }
        .typing { color: #9ca3af; font-style: italic; }

        #controls {
            display: flex; padding: 16px 20px; gap: 12px;
            background: #16213e; border-top: 1px solid #2d3748;
            align-items: center;
        }
        #voice-btn {
            width: 56px; height: 56px; border-radius: 50%;
            background: #3b82f6; border: none; cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            transition: all 0.2s; flex-shrink: 0;
        }
        #voice-btn:hover { background: #2563eb; transform: scale(1.05); }
        #voice-btn.listening { background: #f59e0b; animation: pulse 1s infinite; }
        #voice-btn.processing { background: #6366f1; }
        #voice-btn.disabled { background: #4a5568; cursor: not-allowed; }
        #voice-btn svg { width: 24px; height: 24px; fill: white; }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.08); }
        }

        #input {
            flex: 1; padding: 12px 16px; border-radius: 8px;
            border: 1px solid #4a5568; background: #1a1a2e; color: #eee;
            font-size: 16px;
        }
        #input:focus { outline: none; border-color: #3b82f6; }
        #send {
            padding: 12px 20px; background: #3b82f6; color: white;
            border: none; border-radius: 8px; cursor: pointer; font-weight: 500;
        }
        #send:hover { background: #2563eb; }
        #send:disabled { background: #4a5568; cursor: not-allowed; }

        pre {
            white-space: pre-wrap; word-wrap: break-word;
            font-family: inherit; font-size: inherit; margin: 0;
        }

        #audio-viz {
            height: 4px; background: #2d3748; border-radius: 2px;
            overflow: hidden; margin-top: 8px;
        }
        #audio-viz-bar {
            height: 100%; width: 0%; background: #3b82f6;
            transition: width 0.1s;
        }
    </style>
</head>
<body>
    <div id="header">
        <h1>Voice MCP Agent</h1>
        <span id="status" class="status disconnected">Disconnected</span>
    </div>
    <div id="chat"></div>
    <div id="controls">
        <button id="voice-btn" title="Click to speak">
            <svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1 1.93c-3.94-.49-7-3.85-7-7.93V7h2v1c0 2.76 2.24 5 5 5s5-2.24 5-5V7h2v1c0 4.08-3.06 7.44-7 7.93V19h4v2H8v-2h4v-3.07z"/></svg>
        </button>
        <input id="input" placeholder="Type a message or click mic to speak..." autofocus>
        <button id="send">Send</button>
    </div>
    <div id="audio-viz"><div id="audio-viz-bar"></div></div>

    <!-- VAD Library -->
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js"></script>

    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const send = document.getElementById('send');
        const voiceBtn = document.getElementById('voice-btn');
        const status = document.getElementById('status');
        const vizBar = document.getElementById('audio-viz-bar');

        const sessionId = crypto.randomUUID();
        let ws = null;
        let audioContext = null;
        let isListening = false;
        let isPlaying = false;
        let audioQueue = [];
        let currentSource = null;
        let myvad = null;
        let currentResponse = null;

        // Initialize AudioContext for playback
        function initAudioContext() {
            if (!audioContext) {
                audioContext = new AudioContext({ sampleRate: 24000 });
            }
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
            return audioContext;
        }

        // Play received audio chunk (Int16 PCM @ 24kHz)
        async function playAudioChunk(audioData) {
            const ctx = initAudioContext();

            // Convert Int16 PCM to Float32
            const int16Array = new Int16Array(audioData);
            const float32Array = new Float32Array(int16Array.length);
            for (let i = 0; i < int16Array.length; i++) {
                float32Array[i] = int16Array[i] / 32768.0;
            }

            // Create audio buffer
            const audioBuffer = ctx.createBuffer(1, float32Array.length, 24000);
            audioBuffer.getChannelData(0).set(float32Array);

            // Queue for playback
            audioQueue.push(audioBuffer);
            if (!isPlaying) {
                playNextInQueue();
            }
        }

        function playNextInQueue() {
            if (audioQueue.length === 0) {
                isPlaying = false;
                vizBar.style.width = '0%';
                if (!isListening) updateStatus('connected');
                return;
            }

            isPlaying = true;
            updateStatus('speaking');

            const buffer = audioQueue.shift();
            const source = audioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(audioContext.destination);
            source.onended = playNextInQueue;
            source.start();
            currentSource = source;

            // Visualize playback
            vizBar.style.width = '50%';
        }

        // Stop audio playback (for barge-in)
        function stopPlayback() {
            audioQueue = [];
            if (currentSource) {
                try { currentSource.stop(); } catch(e) {}
                currentSource = null;
            }
            isPlaying = false;
            vizBar.style.width = '0%';
        }

        // Initialize VAD
        async function initVAD() {
            if (myvad) return;

            try {
                myvad = await vad.MicVAD.new({
                    positiveSpeechThreshold: 0.8,
                    negativeSpeechThreshold: 0.5,
                    minSpeechFrames: 3,
                    preSpeechPadFrames: 3,
                    onSpeechStart: () => {
                        console.log('Speech started');
                        stopPlayback();
                        updateStatus('listening');
                        voiceBtn.classList.add('listening');
                        ws.send(JSON.stringify({ type: 'vad_start' }));
                    },
                    onSpeechEnd: (audio) => {
                        console.log('Speech ended, sending audio');
                        updateStatus('processing');
                        voiceBtn.classList.remove('listening');
                        voiceBtn.classList.add('processing');

                        // Convert Float32 to Int16 and send
                        const int16Array = new Int16Array(audio.length);
                        for (let i = 0; i < audio.length; i++) {
                            int16Array[i] = Math.max(-32768, Math.min(32767, audio[i] * 32768));
                        }
                        ws.send(int16Array.buffer);
                        ws.send(JSON.stringify({ type: 'vad_end' }));
                    },
                    onVADMisfire: () => {
                        console.log('VAD misfire');
                        voiceBtn.classList.remove('listening');
                        if (!isPlaying) updateStatus('connected');
                    }
                });
                console.log('VAD initialized');
            } catch (e) {
                console.error('VAD init failed:', e);
                alert('Microphone access required for voice input');
            }
        }

        // Toggle voice mode
        async function toggleVoice() {
            initAudioContext();  // Ensure audio context is ready

            if (isListening) {
                myvad?.pause();
                isListening = false;
                voiceBtn.classList.remove('listening', 'processing');
                if (!isPlaying) updateStatus('connected');
            } else {
                if (!myvad) {
                    await initVAD();
                }
                myvad?.start();
                isListening = true;
                updateStatus('listening');
            }
        }

        function updateStatus(state) {
            status.className = 'status ' + state;
            const labels = {
                connected: 'Connected',
                disconnected: 'Disconnected',
                listening: 'Listening...',
                processing: 'Processing...',
                speaking: 'Speaking...'
            };
            status.textContent = labels[state] || state;
        }

        function connect() {
            const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${wsProtocol}//${location.host}/ws/${sessionId}`);
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {
                updateStatus('connected');
                voiceBtn.classList.remove('disabled');
            };

            ws.onclose = () => {
                updateStatus('disconnected');
                voiceBtn.classList.add('disabled');
                myvad?.pause();
                isListening = false;
                setTimeout(connect, 2000);
            };

            ws.onerror = (e) => {
                console.error('WebSocket error:', e);
            };

            ws.onmessage = (e) => {
                if (e.data instanceof ArrayBuffer) {
                    // Binary audio data
                    playAudioChunk(e.data);
                } else {
                    // JSON message
                    try {
                        const data = JSON.parse(e.data);
                        handleMessage(data);
                    } catch (err) {
                        console.error('Failed to parse message:', err);
                    }
                }
            };
        }

        function handleMessage(data) {
            voiceBtn.classList.remove('processing');

            switch (data.type) {
                case 'transcript':
                    addMessage(data.text, 'user transcript');
                    break;

                case 'response_start':
                    removeTyping();
                    currentResponse = document.createElement('div');
                    currentResponse.className = 'msg assistant';
                    currentResponse.innerHTML = '<pre></pre>';
                    chat.appendChild(currentResponse);
                    break;

                case 'response_text':
                    if (currentResponse) {
                        currentResponse.querySelector('pre').textContent += data.chunk;
                        chat.scrollTop = chat.scrollHeight;
                    }
                    break;

                case 'response_end':
                    currentResponse = null;
                    send.disabled = false;
                    break;

                case 'tool_start':
                    const toolDiv = document.createElement('div');
                    toolDiv.className = 'msg tool';
                    toolDiv.innerHTML = '<span class="tool-name">' + escapeHtml(data.name) + '</span>';
                    chat.appendChild(toolDiv);
                    chat.scrollTop = chat.scrollHeight;
                    break;

                case 'tool_end':
                    const lastTool = chat.querySelector('.tool:last-child');
                    if (lastTool && !lastTool.querySelector('.tool-result')) {
                        const result = document.createElement('div');
                        result.className = 'tool-result';
                        result.innerHTML = '<pre>' + escapeHtml(data.result) + '</pre>';
                        lastTool.appendChild(result);
                    }
                    chat.scrollTop = chat.scrollHeight;
                    break;

                case 'audio_start':
                    updateStatus('speaking');
                    break;

                case 'audio_end':
                    // Status will update when playback finishes
                    break;

                case 'error':
                    removeTyping();
                    addMessage('Error: ' + data.message, 'assistant');
                    send.disabled = false;
                    voiceBtn.classList.remove('processing');
                    break;
            }
        }

        function removeTyping() {
            const typing = document.querySelector('.typing');
            if (typing) typing.remove();
        }

        function addMessage(text, className) {
            const div = document.createElement('div');
            div.className = 'msg ' + className;
            div.innerHTML = '<pre>' + escapeHtml(text) + '</pre>';
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function sendTextMessage() {
            const text = input.value.trim();
            if (!text || !ws || ws.readyState !== 1) return;

            addMessage(text, 'user');
            ws.send(JSON.stringify({ type: 'text', message: text }));
            input.value = '';
            send.disabled = true;

            const typing = document.createElement('div');
            typing.className = 'msg assistant typing';
            typing.textContent = 'Thinking...';
            chat.appendChild(typing);
            chat.scrollTop = chat.scrollHeight;
        }

        // Event listeners
        voiceBtn.onclick = toggleVoice;
        send.onclick = sendTextMessage;
        input.onkeydown = (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendTextMessage();
            }
        };

        // Connect on load
        connect();
    </script>
</body>
</html>
"""


def load_voice_config(config_path: Path) -> VoiceConfig:
    """Load voice configuration from config file with environment variable overrides.

    Environment variables take precedence over config file values:
        VOICE_ENABLED, ASR_DEVICE, TTS_DEVICE, ASR_MODEL, TTS_MODEL,
        TTS_SPEAKER, SAMPLE_RATE_IN, SAMPLE_RATE_OUT
    """
    with open(config_path) as f:
        data = json.load(f)

    voice_data = data.get("voice", {})
    # Support both legacy single "device" and new separate device configs
    default_device = voice_data.get("device", "cuda:0")
    # Normalize "cuda" to "cuda:0" for comparison
    is_default_cuda0 = default_device in ("cuda", "cuda:0")

    # Config file values (with legacy support)
    cfg_asr_device = voice_data.get("asr_device", default_device)
    cfg_tts_device = voice_data.get("tts_device", "cuda:1" if is_default_cuda0 else default_device)

    # Environment variables override config file values
    enabled_str = os.getenv("VOICE_ENABLED", str(voice_data.get("enabled", True)))
    enabled = enabled_str.lower() in ("true", "1", "yes")

    return VoiceConfig(
        enabled=enabled,
        asr_device=os.getenv("ASR_DEVICE", cfg_asr_device),
        tts_device=os.getenv("TTS_DEVICE", cfg_tts_device),
        asr_model=os.getenv("ASR_MODEL", voice_data.get("asr_model", VoiceConfig.asr_model)),
        tts_model=os.getenv("TTS_MODEL", voice_data.get("tts_model", VoiceConfig.tts_model)),
        speaker=os.getenv("TTS_SPEAKER", voice_data.get("speaker", VoiceConfig.speaker)),
        sample_rate_in=int(os.getenv("SAMPLE_RATE_IN", voice_data.get("sample_rate_in", 16000))),
        sample_rate_out=int(os.getenv("SAMPLE_RATE_OUT", voice_data.get("sample_rate_out", 24000))),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent and voice engines on startup."""
    global agent, asr_engine, tts_engine, voice_config

    config_path = Path(app.state.config_path)
    config = Config.from_file(config_path)
    voice_config = load_voice_config(config_path)

    # Initialize MCP Agent
    print("Initializing MCP Agent...")
    agent = MCPAgent(config)
    for server in config.mcp_servers:
        server_type = server.get("type", "stdio")
        name = server.get("name")
        if server_type == "stdio":
            command = server.get("command")
            if command:
                await agent.connect_stdio(command, name)
        elif server_type == "sse":
            url = server.get("url")
            if url:
                await agent.connect_sse(url, name)

    # Initialize Voice Engines if enabled
    if voice_config.enabled:
        print(f"Loading ASR engine: {voice_config.asr_model} on {voice_config.asr_device}")
        from voice.asr import ASREngine
        asr_engine = ASREngine(
            model_id=voice_config.asr_model,
            device=voice_config.asr_device,
        )
        asr_engine.warmup()

        print(f"Loading TTS engine: {voice_config.tts_model} on {voice_config.tts_device}")
        from voice.tts import TTSEngine
        tts_engine = TTSEngine(
            model_id=voice_config.tts_model,
            device=voice_config.tts_device,
            speaker_name=voice_config.speaker,
        )
        tts_engine.warmup()

    tools = agent.get_all_tools()
    print(f"\nVoice Agent ready with {len(tools)} tools:")
    for t in tools:
        print(f"  - {t['function']['name']}")

    yield

    await agent.cleanup()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def home():
    """Serve the voice agent web interface."""
    return HTMLResponse(HTML_CONTENT)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "voice_enabled": voice_config.enabled if voice_config else False,
        "tools": len(agent.get_all_tools()) if agent else 0,
    }


@app.get("/tools")
async def list_tools():
    """List available tools."""
    if not agent:
        return {"tools": []}
    return {
        "tools": [
            {"name": t["function"]["name"], "description": t["function"]["description"]}
            for t in agent.get_all_tools()
        ]
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket handler for voice and text communication."""
    await websocket.accept()

    # Create or get session
    if session_id not in sessions:
        sessions[session_id] = VoiceSession(session_id=session_id)
    session = sessions[session_id]

    try:
        while True:
            message = await websocket.receive()

            # Handle binary audio data
            if "bytes" in message:
                audio_bytes = message["bytes"]
                session.audio_buffer.extend(audio_bytes)
                continue

            # Handle JSON messages
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")

                if msg_type == "vad_start":
                    session.is_speaking = True
                    session.audio_buffer = bytearray()

                elif msg_type == "vad_end":
                    session.is_speaking = False

                    if len(session.audio_buffer) > 0:
                        await process_voice_input(
                            websocket, session, bytes(session.audio_buffer)
                        )
                    session.audio_buffer = bytearray()

                elif msg_type == "text":
                    # Process text input
                    text = data.get("message", "").strip()
                    if text:
                        await process_text_input(websocket, session, text)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Cleanup session after disconnect (optional: keep for reconnect)
        pass


async def process_voice_input(websocket: WebSocket, session: VoiceSession, audio_bytes: bytes):
    """Process voice input: ASR -> Agent -> TTS."""
    global agent, asr_engine, tts_engine

    if not asr_engine:
        await websocket.send_json({"type": "error", "message": "ASR not available"})
        return

    # Convert bytes to numpy array (Int16 PCM from browser @ 16kHz)
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_array.astype(np.float32) / 32768.0

    # Skip very short audio
    if len(audio_float) < voice_config.sample_rate_in * 0.3:  # < 0.3 seconds
        return

    # Transcribe with Whisper
    try:
        transcript = await asr_engine.transcribe_async(audio_float, sample_rate=voice_config.sample_rate_in)
        if not transcript or not transcript.strip():
            return

        # Send transcript to client
        await websocket.send_json({"type": "transcript", "text": transcript})
    except Exception as e:
        await websocket.send_json({"type": "error", "message": f"ASR error: {e}"})
        return

    # Process with agent
    await process_agent_response(websocket, session, transcript)


async def process_text_input(websocket: WebSocket, session: VoiceSession, text: str):
    """Process text input: Agent -> TTS."""
    await process_agent_response(websocket, session, text)


async def process_agent_response(websocket: WebSocket, session: VoiceSession, user_input: str):
    """Run agent and stream TTS audio for response."""
    global agent, tts_engine, voice_config

    if not agent:
        await websocket.send_json({"type": "error", "message": "Agent not ready"})
        return

    # Token queue for TTS
    token_queue: asyncio.Queue = asyncio.Queue()
    response_complete = asyncio.Event()

    async def on_token(token: str, is_final: bool):
        """Callback for each LLM token."""
        if token:
            await websocket.send_json({"type": "response_text", "chunk": token})
        await token_queue.put((token, is_final))
        if is_final:
            response_complete.set()

    async def on_tool_call(event: dict):
        """Callback for tool events."""
        await websocket.send_json(event)

    # Signal response start
    await websocket.send_json({"type": "response_start"})

    # Create token generator for TTS
    async def token_generator():
        while True:
            try:
                token, is_final = await asyncio.wait_for(token_queue.get(), timeout=30.0)
                if token:
                    yield token
                if is_final:
                    break
            except asyncio.TimeoutError:
                break

    # TTS audio callback
    async def on_audio_chunk(audio_bytes: bytes, is_final: bool):
        if audio_bytes:
            await websocket.send_bytes(audio_bytes)

    # Run agent with streaming
    async def run_agent():
        try:
            response, session.history = await agent.run_with_history_streaming(
                user_input,
                session.history,
                on_tool_call=on_tool_call,
                on_token=on_token,
            )
            return response
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
            await token_queue.put(("", True))
            response_complete.set()
            return None

    # Start agent task
    agent_task = asyncio.create_task(run_agent())

    # Signal audio start
    await websocket.send_json({"type": "audio_start"})

    # Stream TTS if available
    if tts_engine and voice_config and voice_config.enabled:
        try:
            await tts_engine.synthesize_streaming(token_generator(), on_audio_chunk)
        except Exception as e:
            print(f"TTS error: {e}")
    else:
        # No TTS - just wait for response completion
        await response_complete.wait()

    # Wait for agent to complete
    response = await agent_task

    # Signal completion
    await websocket.send_json({"type": "audio_end"})
    await websocket.send_json({"type": "response_end", "text": response or ""})


def main():
    parser = argparse.ArgumentParser(description="Voice MCP Agent Web Interface")
    parser.add_argument("--config", "-c", default="mcp_agent_config.json", help="Config file path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", "-p", type=int, default=3000, help="Port to listen on")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config file not found: {args.config}")
        return

    app.state.config_path = args.config
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
