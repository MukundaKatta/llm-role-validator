# llm-role-validator

Validate LLM message role sequences before sending to the API.

Different providers have different rules about message ordering — sending a malformed sequence produces cryptic errors. This library lets you check the sequence first and get a clear, human-readable list of every problem.

## Install

```bash
pip install llm-role-validator
```

## Quick start

```python
from llm_role_validator import validate

messages = [
    {"role": "user", "content": "Hello"},
    {"role": "user", "content": "Still there?"},   # consecutive user messages
    {"role": "assistant", "content": "Yes!"},
]

result = validate(messages, provider="anthropic")
print(result.is_valid)       # False
print(result.summary())
# index 1: consecutive 'user' messages are not allowed
# index 2: last message must be 'user', got 'assistant' (Anthropic requirement)
```

## API

### `validate(messages, *, provider="generic")`

Validate a list of message dicts. Returns a `ValidationResult`.

`provider` accepts `"anthropic"`, `"openai"`, `"gemini"`, `"generic"` (or the `Provider` enum). Unknown strings fall back to `"generic"`.

### `ValidationResult`

| Attribute/Method | Description |
|---|---|
| `is_valid` | `True` when no violations were found |
| `violations` | List of `Violation` objects |
| `by_kind(kind)` | Filter violations by `ViolationKind` |
| `summary()` | Human-readable text summary |
| `to_dict()` | JSON-serialisable representation |

### `Violation`

| Field | Type | Description |
|---|---|---|
| `kind` | `ViolationKind` | Category of violation |
| `index` | `int` | 0-based index of the offending message (`-1` for list-level issues) |
| `description` | `str` | Human-readable explanation |

### `ViolationKind`

| Value | Meaning |
|---|---|
| `unknown_role` | Role string is not recognised |
| `consecutive_same_role` | Two consecutive messages share the same role |
| `system_not_first` | System message is not at index 0 |
| `multiple_system` | More than one system message |
| `empty_messages` | Message list is empty |
| `last_not_user` | Last message is not `user` (Anthropic/Gemini requirement) |
| `first_not_user` | First non-system message is not `user` |
| `missing_role` | Message has no `"role"` key |
| `missing_content` | Message has no `"content"` key |

## Provider rules

| Rule | generic | anthropic | openai | gemini |
|---|---|---|---|---|
| No consecutive same role | ✓ | ✓ | ✓ | ✓ |
| System must be at index 0 | ✓ | ✓ | ✓ | ✓ |
| First non-system must be `user` | — | ✓ | ✓ | ✓ |
| Last message must be `user` | — | ✓ | — | ✓ |
| No system role at all | — | — | — | ✓ |

## License

MIT
