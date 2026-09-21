from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., Any]          # fn(ctx, **kwargs) -> JSON-serialisable

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def obj(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or []}


STR = {"type": "string"}
INT = {"type": "integer"}
NUM = {"type": "number"}
STRS = {"type": "array", "items": {"type": "string"}}
