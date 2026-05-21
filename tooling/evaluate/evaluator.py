"""LLM-driven failure-mode classifier for a single agent_trace.txt."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from ..llm import chat


DEFAULT_MODEL = "qwen3.6"

# Fixed failure taxonomy. The evaluator must pick exactly one of these
# labels; free-form detail goes in `note`. Keep this list short and
# orthogonal so aggregation across many runs stays meaningful.
FAILURE_MODES: list[tuple[str, str]] = [
    ("compile_error_unfixed",
     "Model attempted compiles but could not fix a recurring compile error."),
    ("linker_error",
     "Code compiled but the linker rejected it (undefined references, missing lib)."),
    ("wrong_api_usage",
     "Used non-existent or misnamed methods/types/fields on MMDB or gemmi."),
    ("missing_type_info",
     "Could not locate a needed type / header / symbol despite tool calls."),
    ("bad_construction",
     "Receiver object or input arguments were constructed incorrectly."),
    ("assertion_mismatch",
     "Test/gemmi code compiled and ran but produced values disagreeing with the oracle."),
    ("degenerate_thinking",
     "Stream-abort marker, repetitive thinking, or cyclic reasoning prevented progress."),
    ("ran_out_of_turns",
     "Turn budget exhausted before the agent emitted a final usable artefact."),
    ("gave_up",
     "Agent emitted explicit surrender / 'I cannot' / placeholder output."),
    ("never_compiled",
     "Agent never invoked the compile tool — produced no testable artefact."),
    ("infrastructure",
     "External failure (binary path missing, file I/O, env) not the model's fault."),
    ("other",
     "None of the above fit; describe in note."),
]


SYSTEM_PROMPT = """\
You are a failure-mode analyst for an LLM agent pipeline that ports C++ \
crystallography code from MMDB to gemmi. You will be shown the full trace of \
one agent run that did NOT succeed. Your job is to identify, as quickly as \
possible, the single most likely root cause of failure.

Pick exactly one label from the taxonomy. Provide a 1-3 sentence note \
quoting the most telling evidence from the trace (line numbers or short \
substrings). Be concrete; avoid generic summaries.

Respond ONLY with a single JSON object, no prose around it:

{
  "failure_mode": "<one of the taxonomy labels>",
  "confidence": "low" | "medium" | "high",
  "note": "<1-3 sentences with concrete evidence>",
  "evidence_excerpt": "<one short verbatim quote from the trace, <=200 chars>"
}
"""


_TRACE_HEAD_LINES = 200
_TRACE_TAIL_LINES = 600
_TRACE_LINE_THRESHOLD = _TRACE_HEAD_LINES + _TRACE_TAIL_LINES + 100


def _truncate_trace(text: str) -> tuple[str, bool]:
    """Return (possibly-truncated trace, truncated_flag).

    Long traces are split head + tail so the most useful evidence (early
    setup + final failure) survives without blowing the context window.
    """
    lines = text.splitlines()
    if len(lines) <= _TRACE_LINE_THRESHOLD:
        return text, False
    head = lines[:_TRACE_HEAD_LINES]
    tail = lines[-_TRACE_TAIL_LINES:]
    sep = [
        "",
        f"... [TRUNCATED {len(lines) - _TRACE_HEAD_LINES - _TRACE_TAIL_LINES} lines] ...",
        "",
    ]
    return "\n".join(head + sep + tail), True


def _build_user_prompt(qname: str, stage: str, reason: str,
                       trace: str, truncated: bool) -> str:
    taxonomy = "\n".join(f"  - `{label}`: {desc}" for label, desc in FAILURE_MODES)
    trunc_note = (
        "\n\n(Note: the trace was truncated head+tail to fit context — "
        "middle turns are elided.)" if truncated else ""
    )
    return f"""\
Function: `{qname}`
Failed stage: **{stage}**
Detection reason: {reason}

## Failure-mode taxonomy

{taxonomy}

## Agent trace

```
{trace}
```{trunc_note}
"""


def _extract_json(content: str) -> dict | None:
    content = content.strip()
    # Strip fenced code if present.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        content = m.group(1)
    # Greedy: first { to last }.
    if "{" in content and "}" in content:
        snippet = content[content.index("{"): content.rindex("}") + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None
    return None


@dataclass
class EvaluationResult:
    qname: str
    stage: str
    detection_reason: str
    failure_mode: str
    confidence: str
    note: str
    evidence_excerpt: str
    truncated: bool
    model: str
    raw_response: str

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_trace(*, qname: str, stage: str, detection_reason: str,
                   trace_path: Path, model: str = DEFAULT_MODEL) -> EvaluationResult:
    raw = trace_path.read_text(errors="replace")
    trace, truncated = _truncate_trace(raw)
    user_prompt = _build_user_prompt(qname, stage, detection_reason, trace, truncated)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    resp = chat(messages, model=model, tools=[])
    content = (resp.get("message") or {}).get("content", "") or ""
    parsed = _extract_json(content) or {}

    return EvaluationResult(
        qname=qname,
        stage=stage,
        detection_reason=detection_reason,
        failure_mode=str(parsed.get("failure_mode", "other")),
        confidence=str(parsed.get("confidence", "low")),
        note=str(parsed.get("note", "")).strip(),
        evidence_excerpt=str(parsed.get("evidence_excerpt", "")).strip(),
        truncated=truncated,
        model=model,
        raw_response=content,
    )
