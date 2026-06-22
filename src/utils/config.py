"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def strip_yaml_comment(line: str) -> str:
    quote = None
    for i, ch in enumerate(line):
        if ch in {"'", '"'}:
            quote = None if quote == ch else ch
        elif ch == "#" and quote is None and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def parse_simple_yaml(text: str) -> dict:
    """
    Minimal YAML parser for the repository config shape.

    PyYAML is used when installed; this fallback supports nested mappings,
    block lists, and scalar values so the repo can run without extra setup.
    """
    lines = []
    for raw_line in text.splitlines():
        clean = strip_yaml_comment(raw_line).rstrip()
        if not clean.strip():
            continue
        indent = len(clean) - len(clean.lstrip(" "))
        lines.append((indent, clean.strip()))

    def parse_key_value(content: str) -> tuple[str, Any, bool]:
        if ":" not in content:
            raise ValueError(f"Invalid YAML mapping line: {content}")
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid YAML mapping key: {content}")
        return key, parse_scalar(value) if value else None, bool(value)

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return None, index
        current_indent, content = lines[index]
        if current_indent < indent:
            return None, index
        if content.startswith("- "):
            return parse_list(index, current_indent)
        return parse_dict(index, current_indent)

    def parse_dict(index: int, indent: int) -> tuple[dict, int]:
        result = {}
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation before: {content}")
            if content.startswith("- "):
                break

            key, value, has_inline_value = parse_key_value(content)
            index += 1
            if not has_inline_value and index < len(lines) and lines[index][0] > current_indent:
                value, index = parse_block(index, lines[index][0])
            result[key] = value
        return result, index

    def parse_list(index: int, indent: int) -> tuple[list, int]:
        result = []
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation before: {content}")
            if not content.startswith("- "):
                break

            item_text = content[2:].strip()
            index += 1

            if not item_text:
                value = None
                if index < len(lines) and lines[index][0] > current_indent:
                    value, index = parse_block(index, lines[index][0])
                result.append(value)
                continue

            if ":" in item_text:
                key, value, has_inline_value = parse_key_value(item_text)
                item = {key: value}
                if not has_inline_value and index < len(lines) and lines[index][0] > current_indent:
                    item[key], index = parse_block(index, lines[index][0])
                if index < len(lines) and lines[index][0] > current_indent:
                    extra, index = parse_dict(index, lines[index][0])
                    item.update(extra)
                result.append(item)
            else:
                result.append(parse_scalar(item_text))

        return result, index

    data, index = parse_block(0, lines[0][0] if lines else 0)
    if index != len(lines):
        raise ValueError("Could not parse the complete YAML config")
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def load_yaml_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ModuleNotFoundError:
        return parse_simple_yaml(text)

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data

