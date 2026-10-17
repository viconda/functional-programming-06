"""Вариант 3: неизменяемая родословная как ациклический граф."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class Person:
    """Имя человека и имена известных родителей."""

    name: str
    parents: tuple[str, ...] = ()


Family: TypeAlias = tuple[Person, ...]


def validate(family: Family) -> dict[str, Person]:
    """Проверить уникальность, ссылки на родителей и отсутствие циклов."""
    index = {person.name: person for person in family}
    if len(index) != len(family):
        raise ValueError("Имена должны быть уникальны")
    for person in family:
        if len(person.parents) != len(set(person.parents)):
            raise ValueError("Родитель указан дважды")
        if any(parent not in index for parent in person.parents):
            raise ValueError("Неизвестный родитель")
    visited: set[str] = set()
    active: set[str] = set()

    def visit(name: str) -> None:
        if name in active:
            raise ValueError("В родословной обнаружен цикл")
        if name in visited:
            return
        active.add(name)
        for parent in index[name].parents:
            visit(parent)
        active.remove(name)
        visited.add(name)

    for name in index:
        visit(name)
    return index


def _distances(index: dict[str, Person], name: str, limit: int) -> dict[str, int]:
    if name not in index:
        raise KeyError(f"Неизвестный человек: {name}")
    distances = {name: 0}

    def walk(current: str, depth: int) -> None:
        if depth >= limit:
            return
        for parent in index[current].parents:
            distance = depth + 1
            if distance < distances.get(parent, limit + 1):
                distances[parent] = distance
                walk(parent, distance)

    walk(name, 0)
    return distances


def ancestors(
    family: Family,
    name: str,
    generations: int,
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """Предки до n-го поколения; повторный предок в ближайшем поколении."""
    if generations < 0:
        raise ValueError("Число поколений не может быть отрицательным")
    index = validate(family)
    distances = _distances(index, name, generations)
    return tuple(
        (
            generation,
            tuple(
                sorted(
                    ancestor
                    for ancestor, distance in distances.items()
                    if distance == generation
                )
            ),
        )
        for generation in range(1, min(generations, len(family)) + 1)
        if generation in distances.values()
    )


def common_ancestors(family: Family, first: str, second: str) -> tuple[str, ...]:
    """Ближайшие общие предки по (max расстояний, сумма); все равные."""
    index = validate(family)
    first_distances = _distances(index, first, len(family))
    second_distances = _distances(index, second, len(family))
    shared = first_distances.keys() & second_distances.keys()
    if not shared:
        return ()

    def score(name: str) -> tuple[int, int]:
        one, two = first_distances[name], second_distances[name]
        return max(one, two), one + two

    best = min(map(score, shared))
    return tuple(sorted(name for name in shared if score(name) == best))


def descendants(family: Family, name: str) -> tuple[str, ...]:
    """Вернуть различных потомков, исключив исходного человека."""
    index = validate(family)
    if name not in index:
        raise KeyError(f"Неизвестный человек: {name}")
    children: dict[str, list[str]] = {parent: [] for parent in index}
    for person in family:
        for parent in person.parents:
            children[parent].append(person.name)
    found: set[str] = set()

    def walk(current: str) -> None:
        for child in children[current]:
            if child not in found:
                found.add(child)
                walk(child)

    walk(name)
    return tuple(sorted(found))


def descendant_count(family: Family, name: str) -> int:
    """Подсчитать потомков без повторов по разным ветвям."""
    return len(descendants(family, name))


def sample_family() -> Family:
    """Вернуть учебную родословную с общими предками по двум ветвям."""
    return (
        Person("Алексей"),
        Person("Нина"),
        Person("Олег"),
        Person("Мария"),
        Person("Ирина", ("Алексей", "Нина")),
        Person("Павел", ("Алексей", "Нина")),
        Person("Анна", ("Ирина", "Олег")),
        Person("Дмитрий", ("Ирина", "Олег")),
        Person("Сергей", ("Павел", "Мария")),
        Person("Елена", ("Анна", "Сергей")),
        Person("Пётр"),
    )
