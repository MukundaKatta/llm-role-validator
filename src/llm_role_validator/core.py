"""Validate LLM message role sequences before sending to the API.

Different LLM providers have different rules about message ordering.
Sending a malformed sequence causes cryptic API errors.  This module
lets you validate a message list *before* the call and get a
human-readable description of every problem.

Example::

    from llm_role_validator import validate

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Still there?"},   # consecutive user
        {"role": "assistant", "content": "Yes!"},
    ]

    result = validate(messages, provider="anthropic")
    print(result.is_valid)          # False
    print(result.violations[0].description)
    # index 1: consecutive 'user' messages not allowed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Known message roles across providers."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"  # OpenAI legacy


class Provider(str, Enum):
    """Supported LLM providers with their own validation rules."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    GENERIC = "generic"


class ViolationKind(str, Enum):
    """Category of a rule violation."""

    # Role value is not recognised
    UNKNOWN_ROLE = "unknown_role"
    # Two consecutive messages share the same role
    CONSECUTIVE_SAME_ROLE = "consecutive_same_role"
    # A system message appears somewhere other than index 0
    SYSTEM_NOT_FIRST = "system_not_first"
    # Multiple system messages found
    MULTIPLE_SYSTEM = "multiple_system"
    # The conversation is empty
    EMPTY_MESSAGES = "empty_messages"
    # Last message is not from the user (required by some providers)
    LAST_NOT_USER = "last_not_user"
    # First non-system message is not from the user
    FIRST_NOT_USER = "first_not_user"
    # Message is missing the 'role' field
    MISSING_ROLE = "missing_role"
    # Message is missing the 'content' field
    MISSING_CONTENT = "missing_content"
    # Message is not a mapping (e.g. a string, number, or None)
    INVALID_MESSAGE = "invalid_message"


@dataclass
class Violation:
    """A single rule violation in a message sequence.

    Attributes:
        kind:        Category of violation.
        index:       0-based index of the offending message (``-1`` for
                     list-level violations such as ``EMPTY_MESSAGES``).
        description: Human-readable explanation.
    """

    kind: ViolationKind
    index: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "kind": self.kind.value,
            "index": self.index,
            "description": self.description,
        }

    def __repr__(self) -> str:
        return (
            f"Violation(kind={self.kind.value!r}, index={self.index},"
            f" description={self.description!r})"
        )


@dataclass
class ValidationResult:
    """Outcome of validating a message sequence.

    Attributes:
        violations: All detected :class:`Violation` objects in order.
    """

    violations: list[Violation] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """``True`` when no violations were found."""
        return not self.violations

    def by_kind(self, kind: ViolationKind) -> list[Violation]:
        """Return violations of a specific kind."""
        return [v for v in self.violations if v.kind == kind]

    def summary(self) -> str:
        """Human-readable summary of all violations."""
        if not self.violations:
            return "(valid)"
        lines = [f"index {v.index}: {v.description}" for v in self.violations]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "is_valid": self.is_valid,
            "violations": [v.to_dict() for v in self.violations],
        }

    def __repr__(self) -> str:
        n = len(self.violations)
        return f"ValidationResult(is_valid={self.is_valid}, violations={n})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate(
    messages: list[dict[str, Any]],
    *,
    provider: str | Provider = Provider.GENERIC,
) -> ValidationResult:
    """Validate *messages* for the given *provider*.

    Returns a :class:`ValidationResult`.

    Args:
        messages: List of message dicts, each expected to have at least
                  ``"role"`` and ``"content"`` keys.
        provider: Target provider.  Accepts a :class:`Provider` enum value or
                  a plain string (``"anthropic"``, ``"openai"``, ``"gemini"``,
                  ``"generic"``).  Unknown strings fall back to ``"generic"``.

    Returns:
        A :class:`ValidationResult` listing every violation found.
    """
    # Normalise provider
    if isinstance(provider, str):
        try:
            provider = Provider(provider.lower())
        except ValueError:
            provider = Provider.GENERIC

    violations: list[Violation] = []

    # ---- empty list --------------------------------------------------------
    if not messages:
        violations.append(
            Violation(
                kind=ViolationKind.EMPTY_MESSAGES,
                index=-1,
                description="message list is empty",
            )
        )
        return ValidationResult(violations=violations)

    # ---- per-message checks ------------------------------------------------
    _check_fields(messages, violations)

    # ---- structural checks -------------------------------------------------
    _check_structure(messages, violations)

    # ---- provider-specific checks ------------------------------------------
    if provider == Provider.ANTHROPIC:
        _check_anthropic(messages, violations)
    elif provider in (Provider.OPENAI,):
        _check_openai(messages, violations)
    elif provider == Provider.GEMINI:
        _check_gemini(messages, violations)
    # generic: no extra rules beyond structural checks

    return ValidationResult(violations=violations)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_KNOWN_ROLES = {r.value for r in MessageRole}


def _role_of(msg: Any) -> str | None:
    """Return the role string of *msg*, or ``None`` if absent or unavailable."""
    if not isinstance(msg, dict):
        return None
    return msg.get("role")


def _check_fields(
    messages: list[dict[str, Any]],
    out: list[Violation],
) -> None:
    """Check that each message has 'role' and 'content'."""
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            out.append(
                Violation(
                    kind=ViolationKind.INVALID_MESSAGE,
                    index=i,
                    description=(
                        f"index {i}: message must be a dict, got {type(msg).__name__}"
                    ),
                )
            )
            continue
        if "role" not in msg:
            out.append(
                Violation(
                    kind=ViolationKind.MISSING_ROLE,
                    index=i,
                    description=f"index {i}: message is missing the 'role' field",
                )
            )
        else:
            role_val = msg["role"]
            if role_val not in _KNOWN_ROLES:
                out.append(
                    Violation(
                        kind=ViolationKind.UNKNOWN_ROLE,
                        index=i,
                        description=(
                            f"index {i}: unknown role {role_val!r};"
                            f" expected one of {sorted(_KNOWN_ROLES)}"
                        ),
                    )
                )
        if "content" not in msg:
            out.append(
                Violation(
                    kind=ViolationKind.MISSING_CONTENT,
                    index=i,
                    description=f"index {i}: message is missing the 'content' field",
                )
            )


def _check_structure(
    messages: list[dict[str, Any]],
    out: list[Violation],
) -> None:
    """Provider-agnostic structural checks."""
    system_indices: list[int] = []
    prev_role: str | None = None

    for i, msg in enumerate(messages):
        role = _role_of(msg)
        if role is None:
            prev_role = None
            continue

        # System placement
        if role == MessageRole.SYSTEM.value:
            system_indices.append(i)
            if i != 0:
                out.append(
                    Violation(
                        kind=ViolationKind.SYSTEM_NOT_FIRST,
                        index=i,
                        description=(
                            f"index {i}: 'system' message must be at index 0,"
                            f" found at index {i}"
                        ),
                    )
                )

        # Consecutive same role (excluding system)
        if role == prev_role and role != MessageRole.SYSTEM.value:
            out.append(
                Violation(
                    kind=ViolationKind.CONSECUTIVE_SAME_ROLE,
                    index=i,
                    description=(
                        f"index {i}: consecutive '{role}' messages are not allowed"
                    ),
                )
            )

        prev_role = role

    # Multiple system messages
    if len(system_indices) > 1:
        for idx in system_indices[1:]:
            out.append(
                Violation(
                    kind=ViolationKind.MULTIPLE_SYSTEM,
                    index=idx,
                    description=(
                        f"index {idx}: only one 'system' message is allowed;"
                        f" found additional at index {idx}"
                    ),
                )
            )


def _non_system_messages(
    messages: list[dict[str, Any]],
) -> list[tuple[int, str]]:
    """Return (index, role) pairs for non-system messages with a known role."""
    pairs: list[tuple[int, str]] = []
    for i, msg in enumerate(messages):
        role = _role_of(msg)
        if role and role != MessageRole.SYSTEM.value:
            pairs.append((i, role))
    return pairs


def _check_anthropic(
    messages: list[dict[str, Any]],
    out: list[Violation],
) -> None:
    """Anthropic-specific rules.

    - First non-system message must be 'user'.
    - Last message must be 'user'.
    """
    non_sys = _non_system_messages(messages)
    if not non_sys:
        return

    first_idx, first_role = non_sys[0]
    if first_role != MessageRole.USER.value:
        out.append(
            Violation(
                kind=ViolationKind.FIRST_NOT_USER,
                index=first_idx,
                description=(
                    f"index {first_idx}: first non-system message must be 'user',"
                    f" got {first_role!r} (Anthropic requirement)"
                ),
            )
        )

    last_idx, last_role = non_sys[-1]
    if last_role != MessageRole.USER.value:
        out.append(
            Violation(
                kind=ViolationKind.LAST_NOT_USER,
                index=last_idx,
                description=(
                    f"index {last_idx}: last message must be 'user',"
                    f" got {last_role!r} (Anthropic requirement)"
                ),
            )
        )


def _check_openai(
    messages: list[dict[str, Any]],
    out: list[Violation],
) -> None:
    """OpenAI-specific rules.

    - First non-system message should be 'user'.
    """
    non_sys = _non_system_messages(messages)
    if not non_sys:
        return

    first_idx, first_role = non_sys[0]
    if first_role != MessageRole.USER.value:
        out.append(
            Violation(
                kind=ViolationKind.FIRST_NOT_USER,
                index=first_idx,
                description=(
                    f"index {first_idx}: first non-system message should be 'user',"
                    f" got {first_role!r} (OpenAI recommendation)"
                ),
            )
        )


def _check_gemini(
    messages: list[dict[str, Any]],
    out: list[Violation],
) -> None:
    """Gemini-specific rules.

    - No system messages (Gemini uses a separate system_instruction field).
    - First message must be 'user'.
    - Last message must be 'user'.
    """
    for i, msg in enumerate(messages):
        if _role_of(msg) == MessageRole.SYSTEM.value:
            out.append(
                Violation(
                    kind=ViolationKind.SYSTEM_NOT_FIRST,
                    index=i,
                    description=(
                        f"index {i}: Gemini does not support 'system' role in"
                        f" messages; use system_instruction instead"
                    ),
                )
            )

    non_sys = _non_system_messages(messages)
    if not non_sys:
        return

    first_idx, first_role = non_sys[0]
    if first_role != MessageRole.USER.value:
        out.append(
            Violation(
                kind=ViolationKind.FIRST_NOT_USER,
                index=first_idx,
                description=(
                    f"index {first_idx}: first message must be 'user',"
                    f" got {first_role!r} (Gemini requirement)"
                ),
            )
        )

    last_idx, last_role = non_sys[-1]
    if last_role != MessageRole.USER.value:
        out.append(
            Violation(
                kind=ViolationKind.LAST_NOT_USER,
                index=last_idx,
                description=(
                    f"index {last_idx}: last message must be 'user',"
                    f" got {last_role!r} (Gemini requirement)"
                ),
            )
        )
