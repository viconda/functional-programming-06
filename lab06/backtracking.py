"""Externally pure recursive combinatorial search algorithms."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")
Coordinate = tuple[int, int]


def permutations(xs: Sequence[T]) -> Iterator[tuple[T, ...]]:
    """Yield positional permutations without itertools; duplicates persist."""
    items = tuple(xs)
    if not items:
        yield ()
    for i, value in enumerate(items):
        for tail in permutations(items[:i] + items[i + 1 :]):
            yield (value,) + tail


def combinations(xs: Sequence[T], k: int) -> Iterator[tuple[T, ...]]:
    """Yield positional k-combinations in input order without itertools."""
    if k < 0:
        raise ValueError("k must be nonnegative")
    items = tuple(xs)
    if k == 0:
        yield ()
    elif k <= len(items):
        for i in range(len(items) - k + 1):
            for tail in combinations(items[i + 1 :], k - 1):
                yield (items[i],) + tail


def subset_sum(xs: Sequence[int], target: int) -> Iterator[tuple[int, ...]]:
    """Yield every positional subset whose sum is target, including empty.

    Prune total > target only when all numbers are nonnegative. Signed data
    are searched fully, because a later negative number may reduce total.
    Equal input values at different positions represent different choices.
    """
    items = tuple(xs)
    nonnegative = all(value >= 0 for value in items)

    def search(
        index: int, total: int, chosen: tuple[int, ...]
    ) -> Iterator[tuple[int, ...]]:
        if nonnegative and total > target:
            return
        if index == len(items):
            if total == target:
                yield chosen
            return
        yield from search(index + 1, total, chosen)
        yield from search(index + 1, total + items[index], chosen + (items[index],))

    yield from search(0, 0, ())


def queens_count(n: int, prune: bool = True) -> int:
    """Count n-queen placements with or without early diagonal pruning.

    Both modes assign distinct columns. Without pruning, all n! column
    permutations are generated and diagonals are tested only at a leaf.
    No board or counter is stored outside the current function call.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    if prune:
        full = (1 << n) - 1

        def visit(columns: int, diagonal_left: int, diagonal_right: int) -> int:
            if columns == full:
                return 1
            available = full & ~(columns | diagonal_left | diagonal_right)
            total = 0
            while available:
                bit = available & -available
                available ^= bit
                total += visit(
                    columns | bit,
                    (diagonal_left | bit) << 1,
                    (diagonal_right | bit) >> 1,
                )
            return total

        return visit(0, 0, 0)

    def complete(board: tuple[int, ...], unused: int) -> int:
        if len(board) == n:
            return int(
                all(
                    abs(board[i] - board[j]) != j - i
                    for i in range(n)
                    for j in range(i + 1, n)
                )
            )
        total = 0
        choices = unused
        while choices:
            bit = choices & -choices
            choices ^= bit
            total += complete(board + (bit.bit_length() - 1,), unused ^ bit)
        return total

    return complete((), (1 << n) - 1)


def maze_path(maze: tuple[str, ...]) -> tuple[Coordinate, ...] | None:
    """Find a simple DFS path from the unique S to E, or return None.

    Coordinates are (row, column), zero based. The maze must be rectangular
    and contain only #, ., S, E. This DFS does not promise a shortest path.
    """
    if not maze or not maze[0] or any(len(row) != len(maze[0]) for row in maze):
        raise ValueError("The maze must be a nonempty rectangle")
    if any(char not in "#.SE" for row in maze for char in row):
        raise ValueError("Unknown maze character")
    starts = [
        (r, c)
        for r, row in enumerate(maze)
        for c, char in enumerate(row)
        if char == "S"
    ]
    ends = [
        (r, c)
        for r, row in enumerate(maze)
        for c, char in enumerate(row)
        if char == "E"
    ]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("The maze must contain exactly one S and one E")

    def search(
        cell: Coordinate, visited: frozenset[Coordinate], path: tuple[Coordinate, ...]
    ) -> tuple[Coordinate, ...] | None:
        if cell == ends[0]:
            return path
        row, column = cell
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            neighbor = row + dr, column + dc
            r, c = neighbor
            if (
                0 <= r < len(maze)
                and 0 <= c < len(maze[0])
                and maze[r][c] != "#"
                and neighbor not in visited
            ):
                result = search(neighbor, visited | {neighbor}, path + (neighbor,))
                if result is not None:
                    return result
        return None

    start = starts[0]
    return search(start, frozenset({start}), (start,))
