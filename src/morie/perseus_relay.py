"""Perseus Relay -- serve Perseus as a cloud API endpoint.

Run on Pi (or any machine with Ollama) to let remote users access Perseus
with full tool-calling capabilities over the internet.

Usage:
    python -m morie.perseus_relay                    # default :8421
    python -m morie.perseus_relay --port 9000        # custom port
    python -m morie.perseus_relay --token mysecret   # require auth token

Then from any machine:
    morie percy --cloud https://your-server:8421 "What is Moran's I?"

Or set PERSEUS_CLOUD_URL in .env and it auto-connects.

Security: The relay only exposes Perseus agent capabilities (search, run
functions, read files within sandbox). No shell access, no filesystem
writes outside the project. Optional token auth for production use.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logger = logging.getLogger(__name__)


def _create_agent():
    try:
        from .agent import create_agent
    except ImportError as exc:
        raise RuntimeError(
            "the morie agent is not bundled in this install; run `morie interactive install` to add it"
        ) from exc
    return create_agent()


class PerseusRelayHandler(BaseHTTPRequestHandler):
    agent = None
    auth_token = None

    def do_POST(self):
        if self.path == "/v1/percy":
            self._handle_percy()
        elif self.path == "/v1/health":
            self._respond(200, {"status": "ok", "model": getattr(self.agent, "_model", "unknown")})
        else:
            self._respond(404, {"error": "Not found. Use POST /v1/percy"})

    def do_GET(self):
        if self.path in ("/v1/health", "/health", "/"):
            model = getattr(self.agent, "_model", "provider-chain")
            self._respond(
                200,
                {
                    "status": "ok",
                    "service": "perseus-relay",
                    "model": model,
                    "tools": 12,
                    "functions": "5710+",
                },
            )
        else:
            self._respond(404, {"error": "Not found"})

    def _handle_percy(self):
        if self.auth_token:
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {self.auth_token}":
                self._respond(401, {"error": "Invalid or missing auth token"})
                return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, ValueError):
            self._respond(400, {"error": "Invalid JSON body"})
            return

        question = body.get("question", "")
        if not question:
            self._respond(400, {"error": "Missing 'question' field"})
            return

        start = time.monotonic()
        code, data = answer_question(self.agent, question)
        data["elapsed_s"] = round(time.monotonic() - start, 2)
        self._respond(code, data)

    def _respond(self, code: int, data: dict[str, Any]):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        logger.info(format, *args)


class PerseusCloudClient:
    """Client for connecting to a remote Perseus relay."""

    def __init__(self, url: str, token: str | None = None) -> None:
        self.url = url.rstrip("/")
        self.token = token

    def ask(self, question: str, *, timeout: float = 120.0) -> dict[str, Any]:
        import httpx

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        resp = httpx.post(
            f"{self.url}/v1/percy",
            json={"question": question},
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def health(self, timeout: float = 5.0) -> dict[str, Any]:
        import httpx

        resp = httpx.get(f"{self.url}/v1/health", timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def is_available(self) -> bool:
        try:
            h = self.health()
            return h.get("status") == "ok"
        except Exception:
            return False


def answer_question(agent: Any, question: str) -> tuple[int, dict[str, Any]]:
    """Answer through the local tool-calling agent when it works, otherwise through the
    provider chain (hosted tier, your own endpoint, ...). 503 when nothing answered: a
    relay must never hand a client "LLM request failed" with status 200."""
    if agent is not None:
        try:
            resp = agent.chat(question)
        except Exception as exc:  # the Ollama backend died mid-answer
            logger.info("local agent failed (%s); answering through the provider chain", exc)
        else:
            if str(resp.text or "").strip() and not getattr(resp, "failed", False):
                return 200, {
                    "text": resp.text,
                    "tool_calls": resp.tool_calls_made,
                    "iterations": resp.iterations,
                    "model": resp.model,
                    "backend": "agent",
                }
            logger.info("local agent produced no answer; answering through the provider chain")
    from .perseus import ask_percy

    payload = ask_percy(question)
    text = str(payload.get("output_text") or "")
    if payload.get("mode") == "local_fallback":
        return 503, {
            "error": "no LLM backend reachable: run `morie login` (GitHub, or --email you@example.com) for the hosted tier or start Ollama",
            "text": text,
            "tool_calls": [],
            "iterations": 0,
            "model": str(payload.get("model", "local")),
            "backend": "local_fallback",
        }
    return 200, {
        "text": text,
        "tool_calls": [],
        "iterations": 1,
        "model": str(payload.get("model", "")),
        "backend": str(payload.get("mode", "provider")),
    }


def serve(port: int = 8421, token: str | None = None, bind: str = "127.0.0.1"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    try:
        agent = _create_agent()
    except Exception as exc:  # no agent layer in this install, or no Ollama: provider chain
        agent = None
        logger.info("local tool-calling agent unavailable (%s); answering through the provider chain", exc)
    model_name = getattr(agent, "_model", "provider-chain")
    logger.info("Perseus relay starting on %s:%d with model %s", bind, port, model_name)

    PerseusRelayHandler.agent = agent
    PerseusRelayHandler.auth_token = token

    server = HTTPServer((bind, port), PerseusRelayHandler)
    logger.info('Perseus is online. POST /v1/percy with {"question": "..."}')
    if token:
        logger.info("Auth required: Bearer %s...", token[:4])

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Perseus relay shutting down.")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Perseus Relay Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PERSEUS_PORT", "8421")))
    parser.add_argument("--token", default=os.environ.get("PERSEUS_TOKEN"))
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()
    serve(port=args.port, token=args.token, bind=args.bind)


if __name__ == "__main__":
    main()
