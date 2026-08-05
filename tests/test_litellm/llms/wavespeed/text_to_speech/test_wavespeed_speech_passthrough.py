from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import litellm


class _CaptureHandler(BaseHTTPRequestHandler):
    captured: dict = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _CaptureHandler.captured["body"] = json.loads(body)
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.end_headers()
        self.wfile.write(b"fake-mp3-bytes")

    def log_message(self, format, *args):
        pass


def _start_capture_server() -> tuple[HTTPServer, int]:
    server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def test_minimax_speech_whitelisted_params_reach_wavespeed_shim_body():
    _CaptureHandler.captured.clear()
    server, port = _start_capture_server()
    try:
        litellm.speech(
            model="openai/minimax-speech-2.8-turbo",
            input="hello",
            voice="alloy",
            api_base=f"http://127.0.0.1:{port}/v1",
            api_key="shim-noauth",
            voice_id="female-yujie",
            emotion="happy",
            pitch=2,
            volume=1.5,
            language_boost="Chinese",
            sample_rate=32000,
            bitrate=128000,
            channel="mono",
            pronunciation_dict=["西恩/xi1 en1"],
            english_normalization=True,
            response_format="wav",
            speed=1.2,
        )
    finally:
        server.shutdown()

    body = _CaptureHandler.captured.get("body")
    assert body is not None, "shim never received a request"
    assert body.get("voice_id") == "female-yujie"
    assert body.get("emotion") == "happy"
    assert body.get("pitch") == 2
    assert body.get("volume") == 1.5
    assert body.get("language_boost") == "Chinese"
    assert body.get("sample_rate") == 32000
    assert body.get("bitrate") == 128000
    assert body.get("channel") == "mono"
    assert body.get("pronunciation_dict") == ["西恩/xi1 en1"]
    assert body.get("english_normalization") is True
    assert body.get("response_format") == "wav"
    assert body.get("speed") == 1.2
