"""Backend routing decision (plan section 4.2).

In-process, no network, no model. The slot probe is deliberately NOT here: it is
loopback I/O and sits outside the decision budget, as a pre-flight check.

Order is load-bearing:
  1. privacy veto   - strict => LOCAL only, and never spills (section 4.4 conflict 1)
  2. caller override - subordinate to the veto, logged by the caller
  3. static rule table on (request class, estimated tokens, tools)
"""
from enum import Enum


class Backend(str, Enum):
    LOCAL = "local"
    OPUS = "opus"
    FABLE = "fable"
    FUGU = "fugu"
    JEV = "jev"


class Privacy(str, Enum):
    STRICT = "strict"    # never leaves the machine; fails rather than falls back
    NORMAL = "normal"


class Class(str, Enum):
    TOOL_CALL = "tool_call"
    REASONING = "reasoning"
    INTERACTIVE = "interactive"
    BULK = "bulk"
    CLASSIFY = "classify"
    ESCALATION = "escalation"


# Local per-slot context ceiling. MUST be set from llama-server's reported
# n_ctx_slot at startup, not from the 27k figure in the docs -- see plan 4.5 / R1.
LOCAL_CTX_CEILING = 12288

# Reserve for the response; a request is only local if prompt + this still fits.
RESPONSE_RESERVE = 1024

_RULES = {
    Class.TOOL_CALL:   Backend.LOCAL,
    Class.BULK:        Backend.LOCAL,
    Class.REASONING:   Backend.OPUS,
    Class.INTERACTIVE: Backend.FABLE,
    Class.CLASSIFY:    Backend.JEV,
    Class.ESCALATION:  Backend.FUGU,
}

# Classes that can be served locally at all. Anything else must leave the machine,
# so under a strict privacy veto it has no legal backend.
_LOCAL_CAPABLE = frozenset({Class.TOOL_CALL, Class.BULK, Class.REASONING,
                            Class.INTERACTIVE, Class.CLASSIFY})


class NoLegalBackend(Exception):
    """Privacy veto left no route. Deliberately fails instead of falling back."""


def route(req_class, privacy, est_tokens, has_tools=False, override=None):
    """Return a Backend. Pure function, no I/O."""
    fits_local = (est_tokens + RESPONSE_RESERVE) <= LOCAL_CTX_CEILING

    # 1. privacy veto, evaluated first and beats everything including override
    if privacy is Privacy.STRICT:
        if req_class in _LOCAL_CAPABLE and fits_local:
            return Backend.LOCAL
        raise NoLegalBackend(
            f"{req_class.value}: strict privacy and "
            + ("context exceeds local ceiling" if not fits_local else "class not local-capable")
        )

    # 2. caller override
    if override is not None:
        return override

    # 3. static table, with the context ceiling as the one dynamic input
    if not fits_local:
        return Backend.OPUS if req_class is Class.REASONING else Backend.FABLE
    return _RULES[req_class]
