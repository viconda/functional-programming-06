"""A small functional database: immutable rows, versioned tuples and query AST.

Predicates, keys, transformations and aggregates supplied by the caller must
be pure. This module never mutates a supplied table or row. Tables have bag
semantics, preserving duplicates unless the caller explicitly removes them.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, TypeVar

R = TypeVar("R", bound=tuple)
S = TypeVar("S", bound=tuple)
Key = str | Callable[[Any], Any]


class Student(NamedTuple):
    """An immutable student record."""

    id: int
    name: str
    age: int
    city: str


class Grade(NamedTuple):
    """An immutable grade record."""

    student_id: int
    course: str
    score: int


class JoinedRow(NamedTuple):
    """A matching pair, with original field names kept under left and right."""

    left: tuple
    right: tuple


class Group(NamedTuple):
    """A group key and an immutable aggregation result."""

    key: Any
    value: Any


STUDENTS = (
    Student(1, "Швачкина Виктория", 19, "Алматы"),
    Student(2, "Айдана", 18, "Алматы"),
    Student(3, "Данияр", 20, "Астана"),
    Student(4, "Мария", 21, "Алматы"),
)
GRADES = (
    Grade(1, "Функциональное программирование", 95),
    Grade(1, "Алгоритмы", 90),
    Grade(2, "Функциональное программирование", 88),
    Grade(3, "Функциональное программирование", 76),
    Grade(4, "Функциональное программирование", 92),
    Grade(4, "Алгоритмы", 96),
)


def _immutable(value: Any) -> None:
    """Reject mutable cells so that a tuple cannot hide a mutable list."""
    if isinstance(value, (str, bytes, int, float, complex, bool, type(None))):
        return
    if isinstance(value, (tuple, frozenset)):
        for item in value:
            _immutable(item)
        return
    raise TypeError("table cells and aggregate results must be immutable")


def _row(row: tuple) -> None:
    """Check NamedTuple shape and recursively immutable field values."""
    if not isinstance(row, tuple) or not hasattr(row, "_fields"):
        raise TypeError("each row must be a NamedTuple")
    _immutable(row)


def _table(table: tuple[R, ...]) -> None:
    """Validate immutable records with the same ordered field names.

    Schema compatibility is structural: separate calls to project create
    separate NamedTuple classes whose records must still be composable.
    Type annotations describe values but are not runtime constraints here.
    """
    if not isinstance(table, tuple):
        raise TypeError("a table must be a tuple")
    for row in table:
        _row(row)
        if row._fields != table[0]._fields:
            raise TypeError("rows in one table must have the same schema")


def _key_function(key: Key) -> Callable[[Any], Any]:
    """Accept a field name or a pure callable as a key."""
    if isinstance(key, str):
        return operator.attrgetter(key)
    if callable(key):
        return key
    raise TypeError("a key must be a field name or a callable")


def select(table: tuple[R, ...], predicate: Callable[[R], bool]) -> tuple[R, ...]:
    """Return rows satisfying predicate in their original order."""
    _table(table)
    return tuple(row for row in table if predicate(row))


def project(table: tuple[R, ...], fields: tuple[str, ...]) -> tuple[tuple, ...]:
    """Return NamedTuple records containing exactly the requested fields."""
    _table(table)
    if not isinstance(fields, tuple) or not all(isinstance(f, str) for f in fields):
        raise TypeError("fields must be a tuple of names")
    if len(set(fields)) != len(fields):
        raise ValueError("projection fields must be unique")
    if table and any(field not in table[0]._fields for field in fields):
        raise ValueError("unknown projection field")
    projected = NamedTuple("Projection", [(field, Any) for field in fields])
    return tuple(projected(*(getattr(row, field) for field in fields)) for row in table)


def order_by(table: tuple[R, ...], key: Key) -> tuple[R, ...]:
    """Return a stable ascending ordering; use a key callable for custom order."""
    _table(table)
    return tuple(sorted(table, key=_key_function(key)))


def insert(table: tuple[R, ...], row: R) -> tuple[R, ...]:
    """Append a row to a new table; existing versions remain valid."""
    _table(table)
    _row(row)
    if table and row._fields != table[0]._fields:
        raise TypeError("new row has a different schema")
    return table + (row,)


def update(
    table: tuple[R, ...],
    predicate: Callable[[R], bool],
    changes: Mapping[str, Any],
) -> tuple[R, ...]:
    """Return rows with constant field replacements for the selected records."""
    _table(table)
    if not isinstance(changes, Mapping):
        raise TypeError("changes must be a field-to-value mapping")
    if table and any(field not in table[0]._fields for field in changes):
        raise ValueError("unknown update field")
    for value in changes.values():
        _immutable(value)
    return tuple(row._replace(**changes) if predicate(row) else row for row in table)


def delete(table: tuple[R, ...], predicate: Callable[[R], bool]) -> tuple[R, ...]:
    """Return the complement of select, preserving original row order."""
    _table(table)
    return tuple(row for row in table if not predicate(row))


def join(
    table1: tuple[R, ...], table2: tuple[S, ...], key1: Key, key2: Key
) -> tuple[JoinedRow, ...]:
    """Return an inner equijoin, including every match for repeated keys."""
    _table(table1)
    _table(table2)
    get1, get2 = _key_function(key1), _key_function(key2)
    return tuple(
        JoinedRow(left, right)
        for left in table1
        for right in table2
        if get1(left) == get2(right)
    )


def group_by(
    table: tuple[R, ...], key: Key, aggregate: Callable[[tuple[R, ...]], Any]
) -> tuple[Group, ...]:
    """Group rows in first-key appearance order and apply a pure aggregate.

    Each key is evaluated once. Keys follow Python dictionary semantics:
    the same NaN object groups together, while distinct NaNs stay separate.
    Every aggregate receives a nonempty immutable tuple, including NaN groups.
    """
    _table(table)
    get = _key_function(key)
    groups: dict[Any, list[R]] = {}
    for row in table:
        groups.setdefault(get(row), []).append(row)
    result = tuple(
        Group(value, aggregate(tuple(rows))) for value, rows in groups.items()
    )
    _immutable(result)
    return result


def compile_predicate(expression: tuple) -> Callable[[Any], bool]:
    """Recursively compile a tuple AST without eval or executing source code.

    Comparisons: eq, ne, gt, ge, lt, le with a field name and literal.
    Boolean operations: and/or with at least two children, not with one.
    A nonexistent field raises AttributeError when a row is evaluated.
    """
    if not isinstance(expression, tuple) or not expression:
        raise ValueError("condition must be a nonempty tuple")
    op, *arguments = expression
    if not isinstance(op, str):
        raise ValueError("condition operator must be a string")
    comparisons = {
        "eq": operator.eq,
        "ne": operator.ne,
        "gt": operator.gt,
        "ge": operator.ge,
        "lt": operator.lt,
        "le": operator.le,
    }
    if op in comparisons:
        if len(arguments) != 2 or not isinstance(arguments[0], str):
            raise ValueError("comparison requires a field name and a literal")
        field, literal = arguments
        _immutable(literal)
        compare = comparisons[op]
        return lambda row: bool(compare(getattr(row, field), literal))
    if op in ("and", "or"):
        if len(arguments) < 2:
            raise ValueError("and/or requires at least two child conditions")
        children = tuple(compile_predicate(child) for child in arguments)
        if op == "and":
            return lambda row: all(child(row) for child in children)
        return lambda row: any(child(row) for child in children)
    if op == "not":
        if len(arguments) != 1:
            raise ValueError("not requires exactly one child condition")
        child = compile_predicate(arguments[0])
        return lambda row: not child(row)
    raise ValueError(f"unknown operator: {op}")
