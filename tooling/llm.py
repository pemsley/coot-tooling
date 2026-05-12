"""
LLM backend abstraction. Supports Ollama and OpenAI (or any OpenAI-compatible endpoint).
Backend selection via CT_BACKEND env var (default: 'ollama').
OpenAI config: .openai/config.json in project root with {"api_key": "...", "base_url": "..."}
(base_url is optional; falls back to OPENAI_API_BASE env var, then to the SDK default).
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Protocol

from . import ollama


_backend_local = threading.local()

# Maximum total wall-clock seconds for a single LLM call before it is retried.
# Set to 0 to disable. Override with CT_LLM_TIMEOUT env var.
LLM_CALL_TIMEOUT = int(os.environ.get("CT_LLM_TIMEOUT", 300))
LLM_MAX_RETRIES  = int(os.environ.get("CT_LLM_MAX_RETRIES", 2))

# How many recent tool results to keep in full in the message history. Older
# tool results are replaced with a one-line stub before each chat() dispatch.
# The agents' tool-result cache means the model can re-call any elided tool
# and get the answer back instantly, so this is loss-less in practice but
# cuts mid-run prompt size by ~5–10× on long agent runs.
TOOL_RESULT_KEEP_LAST_N = int(os.environ.get("CT_TOOL_RESULT_KEEP", 3))


def _elide_stale_tool_results(messages: list[dict],
                              keep_last_n: int = TOOL_RESULT_KEEP_LAST_N) -> None:
    """In-place: stub out tool results older than the last `keep_last_n`.

    Pairs each tool message with the tool_call from the preceding assistant
    message (by position) so the stub can name the tool. Already-stubbed
    results are left untouched. Modifies the caller's list — that's
    intentional: each subsequent turn sees the shrunk history.
    """
    if keep_last_n <= 0:
        return

    pairs: list[tuple[int, str]] = []
    last_calls: list = []
    next_call_pos = 0
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            last_calls = m.get("tool_calls") or []
            next_call_pos = 0
        elif role == "tool":
            name = "?"
            if next_call_pos < len(last_calls):
                name = (last_calls[next_call_pos].get("function") or {}).get("name", "?")
            pairs.append((i, name))
            next_call_pos += 1

    if len(pairs) <= keep_last_n:
        return

    for msg_idx, tool_name in pairs[:-keep_last_n]:
        content = messages[msg_idx].get("content") or ""
        if content.startswith("[elided:"):
            continue
        lines = content.count("\n") + 1 if content else 0
        messages[msg_idx]["content"] = (
            f"[elided: {tool_name} result — {lines} lines, {len(content)} chars. "
            "Stale; re-call this tool if you still need it (the result is cached, "
            "so the re-call returns instantly).]"
        )


class _LLMTimeout(Exception):
    """Raised inside a streaming loop when the wall-clock limit is exceeded."""


def _arm_stream_watchdog(resp, deadline_s: int):
    """Force-close `resp` after deadline_s seconds, regardless of trickle chunks.

    Returns the Timer so the caller can cancel it on clean exit. urlopen's
    `timeout=` is a per-read socket idle timeout — if the server emits a tiny
    chunk every N seconds it never fires. This watchdog provides a hard
    wall-clock cap.
    """
    if deadline_s <= 0:
        return None

    def _kill():
        try:
            resp.close()
        except Exception:
            pass

    t = threading.Timer(deadline_s, _kill)
    t.daemon = True
    t.start()
    return t


# Marker appended to thinking when the streaming guard aborts a turn. The
# turn-level degeneracy check in oracle/agent.py treats this marker as ground
# truth — the streaming guard catches paraphrastic loops that confuse the
# turn-level line-frequency check.
STREAM_ABORT_MARKER = "[STREAM ABORTED: degenerate repetition detected]"


def _streaming_thinking_is_degenerate(thinking: str, window: int = 4000) -> bool:
    """Lightweight streaming check: True if the recent tail keeps repeating itself.

    Triggers when a non-trivial line in the last `window` chars appears ≥5 times.
    Tighter and faster than the turn-level check because the goal is to abort a
    runaway stream within one or two check intervals rather than catching every
    edge case. The turn-level check in oracle/agent.py is the safety net.
    """
    tail = thinking[-window:]
    if len(tail) < 1500:
        return False
    lines = [l.strip() for l in tail.splitlines() if len(l.strip()) >= 20]
    if len(lines) < 12:
        return False
    from collections import Counter
    _, top_count = Counter(lines).most_common(1)[0]
    return top_count >= 5


# ── sampling parameters ────────────────────────────────────────────────────────

# Streaming-time degeneracy guard: every N chars of accumulated thinking, scan
# the recent tail for repetition; abort the stream early if found. Catches the
# 1000-line cyclic-thinking failure mode within ~2000 tokens instead of ~8000+.
_DEGENERATE_CHECK_INTERVAL = 2000

SAMPLING_PARAMS = {
    "temperature": float(os.environ.get("CT_OLLAMA_TEMP", 1.0)),
    "top_p": float(os.environ.get("CT_OLLAMA_TOP_P", 0.95)),
    "top_k": int(os.environ.get("CT_OLLAMA_TOP_K", 20)),
    "min_p": float(os.environ.get("CT_OLLAMA_MIN_P", 0.0)),
    "presence_penalty": float(os.environ.get("CT_OLLAMA_PRESENCE_PENALTY", 1.5)),
    # repeat_penalty 1.0 = off, ollama default = 1.1. We use 1.15: enough to
    # discourage the cyclic "Let me write the code now / Actually..." loop
    # without distorting legitimate code-symbol repetition.
    "repeat_penalty": float(os.environ.get("CT_OLLAMA_REPEAT_PENALTY", 1.2)),
    # The penalty only sees the last N tokens. Observed cycles span ~300-400
    # tokens and recur for thousands of tokens, so a 256-token window is blind
    # to them. 2048 gives the penalty a realistic shot at suppressing them.
    "repeat_last_n": int(os.environ.get("CT_OLLAMA_REPEAT_LAST_N", 2048)),
}

OLLAMA_CONTEXT_TOKENS = int(os.environ.get("CT_OLLAMA_NUM_CTX", 32768))
OLLAMA_MAX_TOKENS = int(os.environ.get("CT_OLLAMA_NUM_PREDICT", -1))

# OpenAI host management. Multiple base_urls let parallel workers shard across
# OpenAI-compatible endpoints (e.g. self-hosted vLLM / Ollama replicas).
# Default points at the local Ollama OpenAI-compat endpoint on 11434.
# NOTE: include the `/v1` suffix — the SDK appends `/chat/completions` directly.
_DEFAULT_OPENAI_HOSTS = "http://127.0.0.1:11434/v1"
_openai_hosts_env = os.environ.get("OPENAI_HOSTS", _DEFAULT_OPENAI_HOSTS).strip()
OPENAI_HOSTS: list[str] = [h.strip() for h in _openai_hosts_env.split(",") if h.strip()]
OPENAI_CONFIG_PATH = Path.cwd() / ".openai" / "config.json"


def _load_openai_config() -> dict:
    """Load OpenAI config from .openai/config.json. Returns {"api_key": "...", "base_url": "..."}.

    api_key may instead be supplied via the OPENAI_API_KEY env var; in that case
    the config file is optional.
    """
    if not OPENAI_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(OPENAI_CONFIG_PATH.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {OPENAI_CONFIG_PATH}: {e}")


def set_openai_host(host: str) -> None:
    _backend_local.openai_host = host


def get_openai_host() -> str | None:
    if hasattr(_backend_local, "openai_host"):
        return _backend_local.openai_host
    return OPENAI_HOSTS[0] if OPENAI_HOSTS else None


# ── Backend Protocol ───────────────────────────────────────────────────────────

class Backend(Protocol):
    def chat(self, messages: list[dict], model: str, tools: list[dict]) -> dict:
        """
        Send a chat request to the LLM. Returns a normalized response dict:
        {
            "message": {
                "role": "assistant",
                "thinking": str,
                "content": str,
                "tool_calls": list,
            }
        }
        """
        ...


# ── Ollama Backend ─────────────────────────────────────────────────────────────

class OllamaBackend:
    def chat(self, messages: list[dict], model: str, tools: list[dict]) -> dict:
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": True,
            "think": True,
            "options": {
                "num_ctx": OLLAMA_CONTEXT_TOKENS,
                "num_predict": OLLAMA_MAX_TOKENS,
                **SAMPLING_PARAMS,
            },
        }).encode()

        req = urllib.request.Request(
            ollama.chat_url(),
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        accumulated_thinking = ""
        accumulated_content = ""
        accumulated_tool_calls: list = []
        last_check_len = 0
        prompt_tokens = None
        output_tokens = None
        start_time = time.monotonic()

        resp = urllib.request.urlopen(req, timeout=600)
        watchdog = _arm_stream_watchdog(resp, LLM_CALL_TIMEOUT)
        try:
            for raw_line in resp:
                if LLM_CALL_TIMEOUT > 0:
                    elapsed = time.monotonic() - start_time
                    if elapsed > LLM_CALL_TIMEOUT:
                        raise _LLMTimeout(
                            f"Ollama call exceeded {LLM_CALL_TIMEOUT}s "
                            f"({elapsed:.0f}s elapsed)"
                        )
                line = raw_line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                msg_chunk = chunk.get("message", {})
                accumulated_thinking += msg_chunk.get("thinking", "") or ""
                accumulated_content += msg_chunk.get("content", "") or ""
                if msg_chunk.get("tool_calls"):
                    accumulated_tool_calls = msg_chunk["tool_calls"]
                if chunk.get("done"):
                    prompt_tokens = chunk.get("prompt_eval_count")
                    output_tokens = chunk.get("eval_count")
                    break
                # Streaming degeneracy guard — periodic-check the thinking tail.
                # Only thinking is checked: code/content blocks legitimately
                # contain repeated patterns (parallel EXPECT_EQ, array literals).
                if (len(accumulated_thinking) - last_check_len
                        >= _DEGENERATE_CHECK_INTERVAL):
                    last_check_len = len(accumulated_thinking)
                    if _streaming_thinking_is_degenerate(accumulated_thinking):
                        accumulated_thinking += f"\n\n{STREAM_ABORT_MARKER}\n"
                        break
        except (urllib.error.URLError, ValueError, OSError) as e:
            elapsed = time.monotonic() - start_time
            if LLM_CALL_TIMEOUT > 0 and elapsed >= LLM_CALL_TIMEOUT - 1:
                raise _LLMTimeout(
                    f"Ollama call exceeded {LLM_CALL_TIMEOUT}s "
                    f"(watchdog closed socket after {elapsed:.0f}s)"
                ) from e
            raise
        finally:
            if watchdog is not None:
                watchdog.cancel()
            try:
                resp.close()
            except Exception:
                pass

        return {
            "message": {
                "role": "assistant",
                "thinking": accumulated_thinking,
                "content": accumulated_content,
                "tool_calls": accumulated_tool_calls,
            },
            "_meta": {
                "elapsed_s": time.monotonic() - start_time,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "backend": "ollama",
            },
        }


# ── OpenAI Backend ─────────────────────────────────────────────────────────────

class OpenAIBackend:
    """OpenAI / OpenAI-compatible backend using the official `openai` SDK.

    Picks api_key from .openai/config.json or OPENAI_API_KEY. base_url comes
    from (in order): set_openai_host(), OPENAI_HOSTS env var, config file,
    OPENAI_API_BASE env var, then SDK default (api.openai.com).
    """

    def __init__(self) -> None:
        try:
            from openai import OpenAI  # noqa: F401  (import test)
        except ImportError as e:
            raise ImportError(
                "OpenAI backend requires the `openai` package. Install with: pip install openai"
            ) from e

    def chat(self, messages: list[dict], model: str, tools: list[dict]) -> dict:
        from openai import OpenAI, APIError, APITimeoutError

        config = _load_openai_config()
        # Ollama's OpenAI-compat endpoint ignores the key but the SDK still
        # requires a non-empty string, so fall back to a placeholder.
        api_key = (
            config.get("api_key")
            or os.environ.get("OPENAI_API_KEY")
            or "ollama"
        )

        base_url = (
            get_openai_host()
            or config.get("base_url")
            or os.environ.get("OPENAI_API_BASE")
            or None
        )

        # Per-call timeout. The SDK aborts the streaming connection if no bytes
        # arrive within `timeout` seconds, which subsumes the urllib watchdog.
        sdk_timeout = LLM_CALL_TIMEOUT if LLM_CALL_TIMEOUT > 0 else 600
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=sdk_timeout)

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": SAMPLING_PARAMS["temperature"],
            "top_p": SAMPLING_PARAMS["top_p"],
        }
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": t.get("function", t)} for t in tools
            ]
        if OLLAMA_MAX_TOKENS > 0:
            kwargs["max_tokens"] = OLLAMA_MAX_TOKENS

        # Forward non-standard knobs to the upstream server. Ollama's OpenAI-compat
        # layer accepts `think: true` to enable reasoning output for qwen3 / gpt-oss
        # / deepseek-r1, mirroring the native /api/chat `think` flag we used before.
        # Harmless on real OpenAI (server ignores unknown fields). Disable with
        # CT_THINK=0 (or the --no-thinking CLI flag, which sets it).
        if os.environ.get("CT_THINK", "1") != "0":
            kwargs["extra_body"] = {"think": True}

        accumulated_content = ""
        accumulated_thinking = ""
        tool_calls_map: dict[int, dict] = {}
        prompt_tokens = None
        output_tokens = None
        start_time = time.monotonic()

        try:
            stream = client.chat.completions.create(**kwargs)
        except APIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

        try:
            for chunk in stream:
                if LLM_CALL_TIMEOUT > 0:
                    elapsed = time.monotonic() - start_time
                    if elapsed > LLM_CALL_TIMEOUT:
                        raise _LLMTimeout(
                            f"OpenAI call exceeded {LLM_CALL_TIMEOUT}s ({elapsed:.0f}s elapsed)"
                        )
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens)
                    output_tokens = getattr(usage, "completion_tokens", output_tokens)
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if getattr(delta, "content", None):
                    accumulated_content += delta.content

                # Reasoning/thinking tokens live on non-standard fields:
                #   - DeepSeek / Ollama qwen3:        reasoning_content
                #   - vLLM / some others:             reasoning
                # The SDK keeps unknown fields in model_extra.
                extra = getattr(delta, "model_extra", None) or {}
                rc = (
                    getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", None)
                    or extra.get("reasoning_content")
                    or extra.get("reasoning")
                )
                if rc:
                    accumulated_thinking += rc

                tcs = getattr(delta, "tool_calls", None)
                if tcs:
                    for tc in tcs:
                        idx = tc.index if tc.index is not None else 0
                        slot = tool_calls_map.setdefault(
                            idx, {"type": "function", "function": {"name": "", "arguments": ""}}
                        )
                        if getattr(tc, "id", None):
                            slot["id"] = tc.id
                        if tc.function is not None:
                            if tc.function.name:
                                slot["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                slot["function"]["arguments"] += tc.function.arguments
        except APITimeoutError as e:
            elapsed = time.monotonic() - start_time
            raise _LLMTimeout(
                f"OpenAI call exceeded {LLM_CALL_TIMEOUT}s "
                f"(SDK timeout after {elapsed:.0f}s)"
            ) from e
        finally:
            try:
                stream.close()
            except Exception:
                pass

        final_tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map)]

        # Fallback: some compat layers leave `<think>...</think>` inline in
        # content instead of routing it to a reasoning field.
        if not accumulated_thinking and "<think>" in accumulated_content:
            import re as _re
            think_blocks = _re.findall(
                r"<think>(.*?)</think>", accumulated_content, flags=_re.DOTALL
            )
            if think_blocks:
                accumulated_thinking = "\n".join(b.strip() for b in think_blocks)
                accumulated_content = _re.sub(
                    r"<think>.*?</think>", "", accumulated_content, flags=_re.DOTALL
                ).strip()

        return {
            "message": {
                "role": "assistant",
                "thinking": accumulated_thinking,
                "content": accumulated_content,
                "tool_calls": final_tool_calls,
            },
            "_meta": {
                "elapsed_s": time.monotonic() - start_time,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "backend": "openai",
            },
        }


# ── Backend Selection ──────────────────────────────────────────────────────────

_backend_impl: Backend | None = None


def _get_backend_impl() -> Backend:
    global _backend_impl
    if _backend_impl is None:
        backend_name = os.environ.get("CT_BACKEND", "ollama").lower()
        if backend_name == "openai":
            _backend_impl = OpenAIBackend()
        else:
            _backend_impl = OllamaBackend()
    return _backend_impl


def chat(messages: list[dict], model: str, tools: list[dict]) -> dict:
    """Dispatch to the selected backend, retrying on timeout."""
    _elide_stale_tool_results(messages)
    backend = _get_backend_impl()
    for attempt in range(1 + LLM_MAX_RETRIES):
        try:
            return backend.chat(messages, model, tools)
        except _LLMTimeout as exc:
            if attempt < LLM_MAX_RETRIES:
                print(
                    f"\n  [llm] {exc} — retrying (attempt {attempt + 1}/{LLM_MAX_RETRIES})",
                    flush=True,
                )
            else:
                print(f"\n  [llm] {exc} — all retries exhausted, re-raising", flush=True)
                raise
    raise RuntimeError("unreachable")
