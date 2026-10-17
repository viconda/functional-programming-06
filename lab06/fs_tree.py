"""Чистые рекурсивные запросы к файловой системе из вложенных кортежей."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import PurePosixPath
from typing import Literal, TypeAlias

File: TypeAlias = tuple[Literal["file"], str, int]
Directory: TypeAlias = tuple[Literal["dir"], str, tuple["Entry", ...]]
Entry: TypeAlias = File | Directory


def total_size(entry: Entry) -> int:
    """Рекурсивно просуммировать размеры всех файлов в байтах."""
    match entry:
        case ("file", _, size):
            return size
        case ("dir", _, children):
            return sum(total_size(child) for child in children)
    raise ValueError("Неизвестный вид узла")


def files(entry: Entry, parent: str = "") -> Iterator[tuple[str, int]]:
    """Выдать полные пути от корня модели и размеры файлов."""
    path = f"{parent}/{entry[1]}" if parent else f"/{entry[1]}"
    match entry:
        case ("file", _, size):
            yield path, size
        case ("dir", _, children):
            for child in children:
                yield from files(child, path)
        case _:
            raise ValueError("Неизвестный вид узла")


def find_extension(entry: Entry, extension: str) -> Iterator[str]:
    """Найти файлы по последнему суффиксу без учёта регистра."""
    wanted = extension.lower()
    if wanted and not wanted.startswith("."):
        wanted = "." + wanted
    for path, _ in files(entry):
        if PurePosixPath(path).suffix.lower() == wanted:
            yield path


def largest_file(entry: Entry) -> tuple[str, int] | None:
    """Найти самый большой файл; равенство разрешать порядком обхода."""
    return max(files(entry), key=lambda item: item[1], default=None)


def tree_text(entry: Entry, level: int = 0) -> str:
    """Вернуть строку дерева с отступами, не выполняя вывод."""
    indent = "  " * level
    match entry:
        case ("file", name, size):
            return f"{indent}{name} ({size} байт)"
        case ("dir", name, children):
            lines = (tree_text(child, level + 1) for child in children)
            return "\n".join((f"{indent}{name}/", *lines))
    raise ValueError("Неизвестный вид узла")


def extension_report(entry: Entry) -> dict[str, tuple[int, int]]:
    """Вернуть расширение -> (число файлов, суммарное число байт)."""
    match entry:
        case ("file", name, size):
            return {PurePosixPath(name).suffix.lower(): (1, size)}
        case ("dir", _, children):
            result: dict[str, tuple[int, int]] = {}
            for child in children:
                for suffix, (count, size) in extension_report(child).items():
                    old_count, old_size = result.get(suffix, (0, 0))
                    result[suffix] = (old_count + count, old_size + size)
            return result
    raise ValueError("Неизвестный вид узла")


def sample_tree() -> Directory:
    """Вернуть неизменяемую четырёхуровневую модель для демонстрации."""
    return (
        "dir",
        "project",
        (
            ("file", "README.md", 1200),
            ("file", "LICENSE", 500),
            (
                "dir",
                "src",
                (
                    ("file", "main.py", 2400),
                    ("dir", "utils", (("file", "helpers.PY", 1600),)),
                ),
            ),
            (
                "dir",
                "data",
                (
                    ("file", "people.json", 3000),
                    ("file", "scores.csv", 1800),
                    ("file", "empty.txt", 0),
                ),
            ),
            (
                "dir",
                "docs",
                (
                    ("file", "report.pdf", 8500),
                    ("file", "notes.md", 700),
                ),
            ),
            ("dir", "empty", ()),
        ),
    )
