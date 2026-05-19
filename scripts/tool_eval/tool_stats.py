from __future__ import annotations

import json
import re
from typing import Any, Dict, List


_CALL_RE = re.compile(r"Tool Call:\s*(?P<name>\w+)\(")
_DET_CALL_RE = re.compile(r"Deterministic .* Tool Call:\s*(?P<name>\w+)\(")


def _safe_parse_json_maybe(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_json_substring(s: str) -> str | None:
    # Grab the first {...} block (single-line tool_result JSON).
    m = re.search(r"\{.*\}", s)
    return m.group(0) if m else None


def extract_tool_stats_from_traces(
    results: List[Dict[str, Any]],
    tool_names: List[str],
) -> Dict[str, Any]:
    """
    Parse agent_quick_test.py's reasoning_trace:
      - "... Tool Call: tool_name({...})"
      - "... Tool Result: {json...}"
      - "Deterministic <X> Tool Call: tool_name({...})"
      - "Deterministic <X> Tool Result: {json...}"
    """
    tool_calls = {t: 0 for t in tool_names}
    tool_success = {t: 0 for t in tool_names}
    tool_failure = {t: 0 for t in tool_names}
    samples_with_any_tool = 0
    total_tool_call_events = 0

    for r in results:
        trace = r.get("reasoning_trace") or []
        seen_this_sample = False

        last_called_tool = None

        for line in trace:
            m = _CALL_RE.search(line) or _DET_CALL_RE.search(line)
            if m:
                last_called_tool = m.group("name")
                if last_called_tool in tool_calls:
                    tool_calls[last_called_tool] += 1
                    total_tool_call_events += 1
                    seen_this_sample = True
                continue

            if "Tool Result:" in line or "Deterministic " in line and " Tool Result:" in line:
                if not last_called_tool or last_called_tool not in tool_calls:
                    continue
                json_sub = _extract_json_substring(line)
                parsed = _safe_parse_json_maybe(json_sub) if json_sub else None
                status = None
                if isinstance(parsed, dict):
                    status = parsed.get("status")
                if status is None:
                    # fallback by substring
                    status = "success" if '"status": "success"' in line else "error"

                if status == "success":
                    tool_success[last_called_tool] += 1
                else:
                    tool_failure[last_called_tool] += 1

        if seen_this_sample:
            samples_with_any_tool += 1

    return {
        "tool_call_rate": (samples_with_any_tool / len(results)) if results else 0.0,
        "avg_tool_calls_per_sample": (total_tool_call_events / len(results)) if results else 0.0,
        "by_tool": {
            t: {
                "calls": tool_calls[t],
                "success": tool_success[t],
                "failure": tool_failure[t],
                "exec_success_rate": (tool_success[t] / tool_calls[t]) if tool_calls[t] else 0.0,
            }
            for t in tool_names
        },
    }

