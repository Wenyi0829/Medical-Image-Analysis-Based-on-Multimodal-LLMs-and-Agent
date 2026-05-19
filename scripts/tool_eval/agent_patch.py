from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Set


def _prompt_for_enabled_tools(
    tool_schemas: List[Dict[str, Any]],
    enabled: Set[str],
) -> str:
    if not enabled:
        return (
            "You are a medical VQA agent with NO TOOLS available.\n"
            "Answer the user's question directly in plain text.\n"
            "Do NOT output any <tool_calls> or <tool_response> tags.\n"
        )

    tool_descriptions = ""
    for schema in tool_schemas:
        name = schema.get("name")
        if name not in enabled:
            continue

        params = schema.get("parameters", {}).get("properties", {}) or {}
        required = schema.get("parameters", {}).get("required", []) or []

        param_lines = ""
        for pname, pinfo in params.items():
            req_flag = pname in required
            param_lines += (
                f"      - {pname} ({pinfo.get('type', 'unknown')}, "
                f"{'required' if req_flag else 'optional'}): {pinfo.get('description', '')}\n"
            )

        tool_descriptions += (
            f"  Tool: {name}\n"
            f"  Description: {schema.get('description', '')}\n"
            f"  Parameters:\n{param_lines}\n"
        )

    # Keep tool decision rules consistent with your agent script,
    # but only mention tools that are actually enabled in this ablation.
    decision_rules = ["## Decision Rules"]
    if "analyze_medical_image" in enabled:
        decision_rules.append(
            "- If the question requires interpreting the medical image, ALWAYS call `analyze_medical_image` first."
        )
    if "search_drug_label" in enabled:
        decision_rules.append(
            "- ONLY call `search_drug_label` when the USER question is explicitly drug/medication-related (drug name, dosage/route/frequency, contraindications, boxed warnings, adverse reactions, interactions, toxicity/overdose, pregnancy/lactation/renal/hepatic dose adjustment in a medication context)."
        )
    if "safety_check_medical_answer" in enabled:
        decision_rules.append(
            "- Before giving the FINAL answer to a patient, call `safety_check_medical_answer` on your draft answer to ensure safety and readability."
        )
    decision_rules.append("- Do not call tools for simple descriptive questions.")

    rules_text = "\n".join(decision_rules) + "\n"

    return (
        "You are a medical VQA agent with access to the following tools.\n\n"
        f"{tool_descriptions}"
        f"{rules_text}"
        "## Drug Answer Requirements\n"
        "- For drug-related questions, final answer MUST include:\n"
        "  1) Brief conclusion (1-3 sentences)\n"
        "  2) Key points from label passages\n"
        "  3) At least 1 source evidence item with source/set_id/effective_time/excerpt.\n\n"
        "## Output Format\n"
        "When you decide to call a tool, output EXACTLY in this format:\n"
        "<tool_calls>{\"tool_calls\":[{\"name\":\"<tool_name>\",\"arguments\":{<args as JSON>}}]}</tool_calls>\n\n"
        "After receiving the tool result (marked <tool_response>), incorporate the findings into your final answer.\n"
        "When you have enough information, output your final answer as plain text with NO <tool_calls> tags.\n"
    )


@contextmanager
def patch_agent_tools(
    agent_module: Any,
    enabled_tool_names: List[str],
):
    """
    Runtime-patch the imported agent module to enable a subset of tools.

    This is used for ablation-based evaluation:
      - full: all tools enabled
      - no_tools: no tools enabled (and deterministic BMI routing is disabled)
    """
    orig_schemas = agent_module.TOOL_SCHEMAS
    orig_schema_by_name = agent_module.TOOL_SCHEMA_BY_NAME
    orig_registry = agent_module.TOOL_REGISTRY
    orig_build_system_prompt: Callable[[], str] = agent_module.build_system_prompt

    enabled_set = set(enabled_tool_names)

    try:
        # Patch schema/registry first so execute_tool + arg validation behave.
        new_schemas = [s for s in orig_schemas if s.get("name") in enabled_set]
        agent_module.TOOL_SCHEMAS = new_schemas
        agent_module.TOOL_SCHEMA_BY_NAME = {s["name"]: s for s in new_schemas}
        agent_module.TOOL_REGISTRY = {k: v for k, v in orig_registry.items() if k in enabled_set}

        # Patch system prompt so the model isn't instructed to call tools not enabled.
        def _build_system_prompt_for_ablation() -> str:
            return _prompt_for_enabled_tools(orig_schemas, enabled_set)

        agent_module.build_system_prompt = _build_system_prompt_for_ablation

        yield
    finally:
        agent_module.TOOL_SCHEMAS = orig_schemas
        agent_module.TOOL_SCHEMA_BY_NAME = orig_schema_by_name
        agent_module.TOOL_REGISTRY = orig_registry
        agent_module.build_system_prompt = orig_build_system_prompt

