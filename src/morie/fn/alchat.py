# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat template rendering (Alammar Ch 6)."""

from ._richresult import RichResult

__all__ = ["alammar_chat_template"]


def alammar_chat_template(turns, template_tokens=None):
    """prompt = concat(role_open + content + role_close per turn).

    ``template_tokens`` maps role -> (open, close). An unknown role is
    refused: rendering it with empty markers would make the turn
    invisible to the model while looking fine in a log.

    Examples
    --------
    >>> out = alammar_chat_template(
    ...     [("user", "hi"), ("assistant", "hello")],
    ...     {"user": ("<u>", "</u>"), "assistant": ("<a>", "</a>")})
    >>> out["prompt"]
    '<u>hi</u><a>hello</a>'
    """
    tt = template_tokens or {
        "system": ("<|system|>\n", "\n"),
        "user": ("<|user|>\n", "\n"),
        "assistant": ("<|assistant|>\n", "\n"),
    }
    parts = []
    for role, content in turns:
        role = str(role)
        if role not in tt:
            raise ValueError(
                f"role {role!r} has no template tokens; rendering it "
                "unmarked would hide the turn from the model.")
        o, c = tt[role]
        parts.append(f"{o}{content}{c}")
    prompt = "".join(parts)
    return RichResult(payload={
        "prompt": prompt, "n_turns": len(list(turns)),
        "estimate": float(len(prompt)), "n": len(list(turns)),
        "method": "Chat template rendering (Alammar Ch 6)"})


def cheatsheet():
    return "alchat: role-token concatenation, unknown roles refused"
