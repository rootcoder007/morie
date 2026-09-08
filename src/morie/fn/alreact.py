# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ReAct agent loop (Yao et al. 2023; Alammar Ch 7)."""

from ._richresult import RichResult

__all__ = ["alammar_react_agent_loop"]


def alammar_react_agent_loop(query, tools, model, max_steps=5):
    """Thought, action, observation, repeated until the model answers.

    ``model`` is context -> {"thought": str, "action": str | None,
    "action_input": ..., "final": str | None}; ``tools`` maps action
    names to callables. An unknown action is an ERROR OBSERVATION fed
    back to the model, not a crash -- that is the loop's recovery
    mechanism. Hitting max_steps without a final answer is reported as
    exhausted, never dressed up as an answer.

    References: Alammar and Grootendorst, Ch 7; Yao et al. (2023).
    """
    if not callable(model):
        raise ValueError("model must be callable context -> step dict.")
    steps = int(max_steps)
    if steps < 1:
        raise ValueError("max_steps must be positive.")
    ctx = [{"query": str(query)}]
    trace = []
    for _ in range(steps):
        step = model(list(ctx))
        thought = step.get("thought")
        if step.get("final") is not None:
            trace.append({"thought": thought, "final": step["final"]})
            return RichResult(payload={
                "answer": step["final"], "trace": trace,
                "steps_used": len(trace), "exhausted": False,
                "estimate": float(len(trace)), "n": len(trace),
                "method": "ReAct loop (Yao et al. 2023)"})
        action = step.get("action")
        if action in tools:
            obs = str(tools[action](step.get("action_input")))
        else:
            obs = (f"ERROR: unknown action {action!r}; available: "
                   f"{sorted(tools)}")
        rec = {"thought": thought, "action": action, "observation": obs}
        trace.append(rec)
        ctx.append(rec)
    return RichResult(payload={
        "answer": None, "trace": trace, "steps_used": len(trace),
        "exhausted": True, "estimate": float(len(trace)), "n": len(trace),
        "method": "ReAct loop (Yao et al. 2023)"})


def cheatsheet():
    return "alreact: thought/action/observation loop, exhaustion reported"
