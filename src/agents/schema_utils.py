from __future__ import annotations

import json
from typing import Any, Dict

from jsonschema import Draft202012Validator


def load_schema(schema_text: str) -> Draft202012Validator:
    schema = json.loads(schema_text)
    return Draft202012Validator(schema)


def validate_json(obj: Dict[str, Any], validator: Draft202012Validator) -> None:
    errors = sorted(validator.iter_errors(obj), key=lambda e: e.path)
    if errors:
        msgs = []
        for e in errors[:10]:
            path = "/".join([str(x) for x in e.path])
            msgs.append(f"{path}: {e.message}")
        raise ValueError("Schema validation failed: " + " | ".join(msgs))
