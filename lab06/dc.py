"""Pure divide-and-conquer algorithms for laboratory work 6."""

from __future__ import annotations

from functools import reduce
from typing import NamedTuple, Sequence, TypeVar

T = TypeVar("T", int, float, str)
Number = int | float


def merge(left: Sequence[T], right: Sequence[T]) -> list[T]:
    """Merge sorted sequences stably without changing either argument."""
    result: list[T] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_sort(xs: Sequence[T]) -> list[T]:
    """Return a stable sorted copy using recursive splitting and merging."""
    if len(xs) < 2:
        return list(xs)
    middle = len(xs) // 2
    return merge(merge_sort(xs[:middle]), merge_sort(xs[middle:]))


def quick_sort(xs: Sequence[T]) -> list[T]:
    """Return a sorted copy; select the median of first/middle/last values."""
    if len(xs) < 2:
        return list(xs)
    pivot = sorted((xs[0], xs[len(xs) // 2], xs[-1]))[1]
    lower = [x for x in xs if x < pivot]
    equal = [x for x in xs if x == pivot]
    upper = [x for x in xs if x > pivot]
    return quick_sort(lower) + equal + quick_sort(upper)


def count_inversions(xs: Sequence[T]) -> int:
    """Count pairs i < j, xs[i] > xs[j] in O(n log n) time."""

    def count(items: Sequence[T]) -> tuple[list[T], int]:
        if len(items) < 2:
            return list(items), 0
        middle = len(items) // 2
        left, left_count = count(items[:middle])
        right, right_count = count(items[middle:])
        merged: list[T] = []
        i = j = 0
        inversions = left_count + right_count
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                inversions += len(left) - i
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inversions

    return count(xs)[1]


def naive_inversions(xs: Sequence[T]) -> int:
    """Count inversions by the independent quadratic definition."""
    return sum(xs[i] > xs[j] for i in range(len(xs)) for j in range(i + 1, len(xs)))


class Subarray(NamedTuple):
    """A nonempty interval xs[start:stop] and its sum."""

    total: Number
    start: int
    stop: int


def _best(*items: Subarray) -> Subarray:
    return max(items, key=lambda item: (item.total, -item.start, -item.stop))


def max_subarray_dc(xs: Sequence[Number]) -> Subarray:
    """Find a maximum nonempty subarray in O(n log n); reject empty input.

    Equal sums prefer the smallest start, then the smallest exclusive stop.
    """
    if not xs:
        raise ValueError("A nonempty sequence is required")

    def solve(lo: int, hi: int) -> Subarray:
        if hi - lo == 1:
            return Subarray(xs[lo], lo, hi)
        middle = (lo + hi) // 2
        left = solve(lo, middle)
        right = solve(middle, hi)
        subtotal: Number = 0
        suffix = Subarray(float("-inf"), middle - 1, middle)
        for i in range(middle - 1, lo - 1, -1):
            subtotal += xs[i]
            suffix = _best(suffix, Subarray(subtotal, i, middle))
        subtotal = 0
        prefix = Subarray(float("-inf"), middle, middle + 1)
        for j in range(middle, hi):
            subtotal += xs[j]
            prefix = _best(prefix, Subarray(subtotal, middle, j + 1))
        crossing = Subarray(suffix.total + prefix.total, suffix.start, prefix.stop)
        return _best(left, right, crossing)

    return solve(0, len(xs))


def kadane_reduce(xs: Sequence[Number]) -> Subarray:
    """Find a maximum subarray in O(n) using reduce and immutable state."""
    if not xs:
        raise ValueError("A nonempty sequence is required")

    def step(
        state: tuple[Subarray, Subarray], item: tuple[int, Number]
    ) -> tuple[Subarray, Subarray]:
        ending, best = state
        index, value = item
        ending = _best(
            Subarray(value, index, index + 1),
            Subarray(ending.total + value, ending.start, index + 1),
        )
        return ending, _best(best, ending)

    first = Subarray(xs[0], 0, 1)
    indexed = ((i, xs[i]) for i in range(1, len(xs)))
    return reduce(step, indexed, (first, first))[1]
