"""Recursive transformations of acyclic lists, tuples and dictionaries."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from numbers import Number
from typing import Any

Path = tuple[str | int, ...]


def deep_flatten(x: Any) -> Iterator[Any]:
    """Yield atoms in traversal order; dictionaries contribute values."""
    if isinstance(x, dict):
        for value in x.values():
            yield from deep_flatten(value)
    elif isinstance(x, (list, tuple)):
        for value in x:
            yield from deep_flatten(value)
    else:
        yield x


def deep_depth(x: Any) -> int:
    """Return container depth: atom 0, empty container 1."""
    if isinstance(x, dict):
        return 1 + max((deep_depth(v) for v in x.values()), default=0)
    if isinstance(x, (list, tuple)):
        return 1 + max((deep_depth(v) for v in x), default=0)
    return 0


def deep_map(f: Callable[[Any], Any], x: Any) -> Any:
    """Apply a pure f to atoms, preserving list/tuple/dict structure."""
    if isinstance(x, list):
        return [deep_map(f, value) for value in x]
    if isinstance(x, tuple):
        return tuple(deep_map(f, value) for value in x)
    if isinstance(x, dict):
        return {key: deep_map(f, value) for key, value in x.items()}
    return f(x)


def deep_sum(x: Any) -> Any:
    """Sum compatible numeric atoms; strings, None and booleans are ignored.

    Python bool is an int subclass, but JSON true/false are logical values.
    Empty structures contribute zero. Numeric atoms must support addition.
    """
    return sum(
        value
        for value in deep_flatten(x)
        if isinstance(value, Number) and not isinstance(value, bool)
    )


def paths(doc: Any) -> Iterator[tuple[Path, Any]]:
    """Yield JSON leaf paths; an empty object/array is a terminal leaf too."""

    def walk(node: Any, prefix: Path) -> Iterator[tuple[Path, Any]]:
        if isinstance(node, dict) and node:
            for key, value in node.items():
                yield from walk(value, prefix + (key,))
        elif isinstance(node, (list, tuple)) and node:
            for index, value in enumerate(node):
                yield from walk(value, prefix + (index,))
        else:
            yield prefix, node

    yield from walk(doc, ())


def _child(doc: Any, key: str | int) -> Any:
    if isinstance(doc, dict):
        return doc[key]
    if isinstance(doc, (list, tuple)):
        if not isinstance(key, int) or isinstance(key, bool):
            raise TypeError("A sequence path component must be an integer")
        if key < 0 or key >= len(doc):
            raise IndexError(key)
        return doc[key]
    raise TypeError("Cannot traverse an atomic value")


def get_path(doc: Any, path: Path) -> Any:
    """Read an existing path; an empty path returns the root value."""
    if not path:
        return doc
    return get_path(_child(doc, path[0]), path[1:])


def set_path(doc: Any, path: Path, value: Any) -> Any:
    """Replace an existing path by copying its containers, sharing the rest.

    The empty path replaces the root. Missing keys and invalid indices raise
    the same errors as reading; this function does not invent containers.
    """
    if not path:
        return value
    key = path[0]
    updated = set_path(_child(doc, key), path[1:], value)
    if isinstance(doc, dict):
        return {k: updated if k == key else v for k, v in doc.items()}
    if isinstance(doc, list):
        return [updated if i == key else v for i, v in enumerate(doc)]
    return tuple(updated if i == key else v for i, v in enumerate(doc))
