"""Tests for llm-role-validator."""

from __future__ import annotations

from llm_role_validator import (
    MessageRole,
    Provider,
    ValidationResult,
    Violation,
    ViolationKind,
    validate,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_message_role_values():
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"
    assert MessageRole.SYSTEM.value == "system"
    assert MessageRole.TOOL.value == "tool"


def test_provider_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.GEMINI.value == "gemini"
    assert Provider.GENERIC.value == "generic"


def test_violation_kind_is_str():
    assert isinstance(ViolationKind.CONSECUTIVE_SAME_ROLE, str)


# ---------------------------------------------------------------------------
# Violation
# ---------------------------------------------------------------------------


def test_violation_to_dict():
    v = Violation(kind=ViolationKind.UNKNOWN_ROLE, index=2, description="bad role")
    d = v.to_dict()
    assert d["kind"] == "unknown_role"
    assert d["index"] == 2
    assert d["description"] == "bad role"


def test_violation_repr():
    v = Violation(kind=ViolationKind.MISSING_ROLE, index=0, description="no role")
    r = repr(v)
    assert "missing_role" in r
    assert "0" in r


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


def test_result_is_valid_when_empty():
    result = ValidationResult(violations=[])
    assert result.is_valid


def test_result_invalid_when_violations():
    v = Violation(kind=ViolationKind.EMPTY_MESSAGES, index=-1, description="empty")
    result = ValidationResult(violations=[v])
    assert not result.is_valid


def test_result_by_kind():
    v1 = Violation(kind=ViolationKind.MISSING_ROLE, index=0, description="x")
    v2 = Violation(kind=ViolationKind.CONSECUTIVE_SAME_ROLE, index=1, description="y")
    result = ValidationResult(violations=[v1, v2])
    assert len(result.by_kind(ViolationKind.MISSING_ROLE)) == 1
    assert len(result.by_kind(ViolationKind.CONSECUTIVE_SAME_ROLE)) == 1
    assert len(result.by_kind(ViolationKind.UNKNOWN_ROLE)) == 0


def test_result_summary_valid():
    result = ValidationResult(violations=[])
    assert result.summary() == "(valid)"


def test_result_summary_with_violations():
    v = Violation(kind=ViolationKind.EMPTY_MESSAGES, index=-1, description="empty")
    result = ValidationResult(violations=[v])
    s = result.summary()
    assert "empty" in s


def test_result_to_dict():
    result = ValidationResult(violations=[])
    d = result.to_dict()
    assert d["is_valid"] is True
    assert d["violations"] == []


def test_result_to_dict_with_violations():
    v = Violation(kind=ViolationKind.MISSING_ROLE, index=0, description="no role")
    result = ValidationResult(violations=[v])
    d = result.to_dict()
    assert not d["is_valid"]
    assert len(d["violations"]) == 1


def test_result_repr():
    result = ValidationResult(violations=[])
    assert "ValidationResult" in repr(result)


# ---------------------------------------------------------------------------
# validate() — empty
# ---------------------------------------------------------------------------


def test_empty_messages():
    result = validate([])
    assert not result.is_valid
    assert len(result.by_kind(ViolationKind.EMPTY_MESSAGES)) == 1


def test_empty_messages_index_is_minus_one():
    result = validate([])
    assert result.violations[0].index == -1


# ---------------------------------------------------------------------------
# validate() — missing fields
# ---------------------------------------------------------------------------


def test_missing_role():
    result = validate([{"content": "hello"}])
    assert len(result.by_kind(ViolationKind.MISSING_ROLE)) == 1


def test_missing_content():
    result = validate([{"role": "user"}])
    assert len(result.by_kind(ViolationKind.MISSING_CONTENT)) == 1


def test_missing_both_fields():
    result = validate([{}])
    kinds = {v.kind for v in result.violations}
    assert ViolationKind.MISSING_ROLE in kinds
    assert ViolationKind.MISSING_CONTENT in kinds


def test_unknown_role():
    result = validate([{"role": "bot", "content": "hi"}])
    assert len(result.by_kind(ViolationKind.UNKNOWN_ROLE)) == 1


def test_unknown_role_description_mentions_known_roles():
    result = validate([{"role": "bot", "content": "hi"}])
    desc = result.violations[0].description
    assert "user" in desc


# ---------------------------------------------------------------------------
# validate() — structural
# ---------------------------------------------------------------------------


def test_valid_single_user():
    result = validate([{"role": "user", "content": "hello"}])
    assert result.is_valid


def test_valid_user_assistant():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you?"},
    ]
    result = validate(messages)
    assert result.is_valid


def test_consecutive_user():
    messages = [
        {"role": "user", "content": "msg1"},
        {"role": "user", "content": "msg2"},
    ]
    result = validate(messages)
    assert len(result.by_kind(ViolationKind.CONSECUTIVE_SAME_ROLE)) == 1
    assert result.violations[0].index == 1


def test_consecutive_assistant():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "assistant", "content": "hi again"},
    ]
    result = validate(messages)
    assert len(result.by_kind(ViolationKind.CONSECUTIVE_SAME_ROLE)) == 1


def test_system_not_first():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "be helpful"},
    ]
    result = validate(messages)
    assert len(result.by_kind(ViolationKind.SYSTEM_NOT_FIRST)) == 1
    assert result.violations[0].index == 1


def test_system_at_index_zero_is_valid():
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    ]
    result = validate(messages)
    assert result.is_valid


def test_multiple_system_messages():
    messages = [
        {"role": "system", "content": "first"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "second"},
    ]
    result = validate(messages)
    assert len(result.by_kind(ViolationKind.MULTIPLE_SYSTEM)) == 1
    assert result.by_kind(ViolationKind.MULTIPLE_SYSTEM)[0].index == 2


def test_no_consecutive_violation_for_system():
    # Two system messages should flag MULTIPLE_SYSTEM but not CONSECUTIVE_SAME_ROLE
    messages = [
        {"role": "system", "content": "a"},
        {"role": "system", "content": "b"},
    ]
    result = validate(messages)
    assert len(result.by_kind(ViolationKind.CONSECUTIVE_SAME_ROLE)) == 0


# ---------------------------------------------------------------------------
# validate() — provider string normalisation
# ---------------------------------------------------------------------------


def test_provider_string_anthropic():
    result = validate([{"role": "user", "content": "hi"}], provider="anthropic")
    assert result.is_valid


def test_provider_string_uppercase():
    result = validate([{"role": "user", "content": "hi"}], provider="ANTHROPIC")
    assert result.is_valid


def test_unknown_provider_string_falls_back_to_generic():
    result = validate([{"role": "user", "content": "hi"}], provider="unknown_provider")
    assert result.is_valid


# ---------------------------------------------------------------------------
# validate() — Anthropic rules
# ---------------------------------------------------------------------------


def test_anthropic_valid():
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "bye"},
    ]
    result = validate(messages, provider="anthropic")
    assert result.is_valid


def test_anthropic_last_not_user():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = validate(messages, provider="anthropic")
    assert len(result.by_kind(ViolationKind.LAST_NOT_USER)) == 1
    assert result.by_kind(ViolationKind.LAST_NOT_USER)[0].index == 1


def test_anthropic_first_not_user():
    messages = [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "hi"},
    ]
    result = validate(messages, provider="anthropic")
    assert len(result.by_kind(ViolationKind.FIRST_NOT_USER)) == 1


def test_anthropic_no_extra_violations_for_valid():
    messages = [{"role": "user", "content": "hi"}]
    result = validate(messages, provider="anthropic")
    assert result.is_valid


# ---------------------------------------------------------------------------
# validate() — OpenAI rules
# ---------------------------------------------------------------------------


def test_openai_valid():
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    result = validate(messages, provider="openai")
    assert result.is_valid


def test_openai_first_not_user():
    messages = [
        {"role": "assistant", "content": "hello first"},
        {"role": "user", "content": "hi"},
    ]
    result = validate(messages, provider="openai")
    assert len(result.by_kind(ViolationKind.FIRST_NOT_USER)) == 1


def test_openai_last_not_user_no_extra_violation():
    # OpenAI does not require last=user
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = validate(messages, provider="openai")
    assert result.is_valid


# ---------------------------------------------------------------------------
# validate() — Gemini rules
# ---------------------------------------------------------------------------


def test_gemini_valid():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "bye"},
    ]
    result = validate(messages, provider="gemini")
    assert result.is_valid


def test_gemini_system_message_flagged():
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
    ]
    result = validate(messages, provider="gemini")
    # Gemini flags system role as SYSTEM_NOT_FIRST at index > 0 OR at index 0
    assert not result.is_valid


def test_gemini_last_not_user():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = validate(messages, provider="gemini")
    assert len(result.by_kind(ViolationKind.LAST_NOT_USER)) == 1


def test_gemini_first_not_user():
    messages = [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "hi"},
    ]
    result = validate(messages, provider="gemini")
    assert len(result.by_kind(ViolationKind.FIRST_NOT_USER)) == 1


# ---------------------------------------------------------------------------
# validate() — Provider enum accepted directly
# ---------------------------------------------------------------------------


def test_provider_enum_accepted():
    messages = [{"role": "user", "content": "hi"}]
    result = validate(messages, provider=Provider.ANTHROPIC)
    assert result.is_valid


# ---------------------------------------------------------------------------
# validate() — tool role
# ---------------------------------------------------------------------------


def test_tool_role_recognised():
    messages = [
        {"role": "user", "content": "run tool"},
        {"role": "tool", "content": "result"},
        {"role": "user", "content": "ok"},
    ]
    result = validate(messages, provider="generic")
    # Tool role consecutive with user is fine (different roles); structural ok
    kinds = {v.kind for v in result.violations}
    assert ViolationKind.UNKNOWN_ROLE not in kinds


# ---------------------------------------------------------------------------
# validate() — malformed (non-dict) messages must not crash
# ---------------------------------------------------------------------------


def test_non_dict_message_string():
    result = validate(["not a dict"])
    assert len(result.by_kind(ViolationKind.INVALID_MESSAGE)) == 1
    assert result.violations[0].index == 0


def test_non_dict_message_none():
    result = validate([None])
    assert len(result.by_kind(ViolationKind.INVALID_MESSAGE)) == 1


def test_non_dict_message_number():
    result = validate([123])
    assert len(result.by_kind(ViolationKind.INVALID_MESSAGE)) == 1


def test_non_dict_message_mixed_with_valid():
    messages = [
        {"role": "user", "content": "hi"},
        5,
        {"role": "user", "content": "bye"},
    ]
    result = validate(messages)
    invalid = result.by_kind(ViolationKind.INVALID_MESSAGE)
    assert len(invalid) == 1
    assert invalid[0].index == 1


def test_non_dict_message_provider_paths_do_not_crash():
    # Each provider branch must tolerate non-dict entries without raising.
    for provider in ("anthropic", "openai", "gemini", "generic"):
        result = validate([None, {"role": "user", "content": "hi"}], provider=provider)
        assert len(result.by_kind(ViolationKind.INVALID_MESSAGE)) == 1


def test_invalid_message_kind_value():
    assert ViolationKind.INVALID_MESSAGE.value == "invalid_message"
