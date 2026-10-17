"""All sixteen practical examples from lecture 06, with input validation.

The list-slicing versions deliberately follow the lecture. They are teaching
implementations for small inputs, not algorithms for unbounded Python lists.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, TypeVar

T = TypeVar("T")
U = TypeVar("U")
A = TypeVar("A")


def natural(value: int, name: str = "n") -> None:
    """Require a nonnegative integer; booleans are not accepted as integers."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def factorial(n: int) -> int:
    """Return n! using linear recursion, including factorial(0) == 1."""
    natural(n)
    return 1 if n == 0 else n * factorial(n - 1)


def factorial_traced(n: int, depth: int = 0) -> int:
    """Print actual nested calls and returns; tracing intentionally uses I/O."""
    natural(n)
    natural(depth, "depth")
    indent = "  " * depth
    print(f"{indent}вызов factorial({n})")
    result = 1 if n == 0 else n * factorial_traced(n - 1, depth + 1)
    print(f"{indent}возврат {result}")
    return result


def depth(n: int) -> int:
    """Create a linear recursive chain to demonstrate the recursion limit."""
    natural(n)
    return 0 if n == 0 else 1 + depth(n - 1)


def fib(n: int) -> int:
    """Return the nth Fibonacci number using the naive recursive definition."""
    natural(n)
    return n if n < 2 else fib(n - 1) + fib(n - 2)


def is_even(n: int) -> bool:
    """Test parity by mutual recursion for nonnegative integers."""
    natural(n)
    return True if n == 0 else is_odd(n - 1)


def is_odd(n: int) -> bool:
    """Test oddness by calling is_even on the smaller problem."""
    natural(n)
    return False if n == 0 else is_even(n - 1)


def factorial_acc(n: int, acc: int = 1) -> int:
    """Return acc * n! with tail recursion; CPython still allocates frames."""
    natural(n)
    if type(acc) is not int:
        raise TypeError("acc must be an integer")
    return acc if n == 0 else factorial_acc(n - 1, acc * n)


def ackermann(m: int, n: int) -> int:
    """Compute Ackermann's nested recursion for small nonnegative inputs."""
    natural(m, "m")
    natural(n)
    if m == 0:
        return n + 1
    if n == 0:
        return ackermann(m - 1, 1)
    return ackermann(m - 1, ackermann(m, n - 1))


def fast_pow(x: int | float, n: int) -> int | float:
    """Return x**n for a nonnegative integer n by repeated squaring."""
    natural(n)
    if n == 0:
        return 1
    if n % 2 == 0:
        half = fast_pow(x, n // 2)
        return half * half
    return x * fast_pow(x, n - 1)


def gcd(a: int, b: int) -> int:
    """Return a nonnegative GCD; gcd(0, 0) follows Python's value of zero."""
    if type(a) is not int or type(b) is not int:
        raise TypeError("a and b must be integers")
    a, b = abs(a), abs(b)
    return a if b == 0 else gcd(b, a % b)


def hanoi(
    n: int, src: str = "A", dst: str = "C", via: str = "B"
) -> list[tuple[int, str, str]]:
    """Return every legal move, preserving the lecture's concatenation form."""
    natural(n)
    if len({src, dst, via}) != 3:
        raise ValueError("peg names must be distinct")
    if n == 0:
        return []
    return hanoi(n - 1, src, via, dst) + [(n, src, dst)] + hanoi(n - 1, via, dst, src)


def binary_search(
    xs: Sequence[T], target: T, lo: int = 0, hi: int | None = None
) -> int:
    """Return an index or -1. The caller supplies an ascending sequence.

    Sorting is a precondition; checking the whole sequence on each lookup
    would hide the logarithmic complexity of this indexed implementation.
    """
    hi = len(xs) - 1 if hi is None else hi
    if not 0 <= lo <= len(xs) or not -1 <= hi < len(xs):
        raise ValueError("search bounds are outside the sequence")
    if lo > hi:
        return -1
    mid = (lo + hi) // 2
    if xs[mid] == target:
        return mid
    if xs[mid] < target:
        return binary_search(xs, target, mid + 1, hi)
    return binary_search(xs, target, lo, mid - 1)


def merge(left: list[T], right: list[T]) -> list[T]:
    """Merge sorted lists recursively, retaining the lecture's copied slices."""
    if not left:
        return list(right)
    if not right:
        return list(left)
    if left[0] <= right[0]:
        return [left[0]] + merge(left[1:], right)
    return [right[0]] + merge(left, right[1:])


def merge_sort(xs: Sequence[T]) -> list[T]:
    """Return a sorted copy; this pedagogical merge makes total time O(n²)."""
    if len(xs) <= 1:
        return list(xs)
    mid = len(xs) // 2
    return merge(merge_sort(xs[:mid]), merge_sort(xs[mid:]))


def quicksort(xs: Sequence[T]) -> list[T]:
    """Return a sorted copy, using the first item as the pivot."""
    if len(xs) <= 1:
        return list(xs)
    pivot, *rest = xs
    smaller = [x for x in rest if x < pivot]
    larger = [x for x in rest if x >= pivot]
    return quicksort(smaller) + [pivot] + quicksort(larger)


def length(xs: Sequence[T]) -> int:
    """Count items by structural pattern matching and head/tail recursion."""
    match xs:
        case []:
            return 0
        case [_, *tail]:
            return 1 + length(tail)
        case _:
            raise TypeError("xs must be a list or tuple-like sequence")


def my_map(function: Callable[[T], U], xs: Sequence[T]) -> list[U]:
    """Transform each item, without changing the input sequence."""
    match xs:
        case []:
            return []
        case [head, *tail]:
            return [function(head)] + my_map(function, tail)
        case _:
            raise TypeError("xs must be a list or tuple-like sequence")


def my_filter(predicate: Callable[[T], bool], xs: Sequence[T]) -> list[T]:
    """Keep items whose predicate is truthy, preserving order."""
    if not xs:
        return []
    head, *tail = xs
    return ([head] if predicate(head) else []) + my_filter(predicate, tail)


def fold_left(function: Callable[[A, T], A], acc: A, xs: Sequence[T]) -> A:
    """Fold from left to right with a tail call and copied sequence tail."""
    return acc if not xs else fold_left(function, function(acc, xs[0]), xs[1:])


def flatten(nested: Sequence[Any]) -> list[Any]:
    """Flatten finite, acyclic nested lists and tuples in left-to-right order."""
    result: list[Any] = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def nested_depth(value: Any) -> int:
    """Return zero for a scalar, one for an empty list, or nesting depth."""
    if not isinstance(value, (list, tuple)):
        return 0
    return 1 + max(map(nested_depth, value), default=0)


class Node(NamedTuple):
    """An immutable binary search tree node."""

    value: int
    left: Node | None = None
    right: Node | None = None


def insert(tree: Node | None, value: int) -> Node:
    """Insert an integer, copying only its path; ignore duplicate keys."""
    if tree is None:
        return Node(value)
    if value < tree.value:
        return tree._replace(left=insert(tree.left, value))
    if value > tree.value:
        return tree._replace(right=insert(tree.right, value))
    return tree


def inorder(tree: Node | None) -> list[int]:
    """Traverse the tree in ascending key order."""
    return (
        [] if tree is None else inorder(tree.left) + [tree.value] + inorder(tree.right)
    )


def height(tree: Node | None) -> int:
    """Return node-level height, using zero for an empty tree."""
    return 0 if tree is None else 1 + max(height(tree.left), height(tree.right))


def contains(tree: Node | None, value: int) -> bool:
    """Search one BST branch at each step."""
    if tree is None:
        return False
    if value == tree.value:
        return True
    return contains(tree.left if value < tree.value else tree.right, value)


def find_values(document: Any, key: str) -> list[Any]:
    """Collect all values under key in an acyclic JSON-like document."""
    if isinstance(document, dict):
        own = [document[key]] if key in document else []
        return own + [
            found for value in document.values() for found in find_values(value, key)
        ]
    if isinstance(document, list):
        return [found for item in document for found in find_values(item, key)]
    return []


def permutations(xs: list[T]) -> list[list[T]]:
    """Enumerate positional permutations; equal input values may repeat."""
    if len(xs) <= 1:
        return [list(xs)]
    return [
        [value] + permutation
        for i, value in enumerate(xs)
        for permutation in permutations(xs[:i] + xs[i + 1 :])
    ]


def subsets(xs: list[T]) -> list[list[T]]:
    """Enumerate positional subsets, including the empty subset."""
    if not xs:
        return [[]]
    rest = subsets(xs[1:])
    return rest + [[xs[0]] + subset for subset in rest]


def queens(n: int, row: int = 0, placed: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    """Return all valid queen placements, rejecting an invalid starting state."""
    natural(n)
    natural(row, "row")
    if row != len(placed) or row > n:
        raise ValueError("row must equal the number of placed queens")
    if any(type(column) is not int or not 0 <= column < n for column in placed):
        raise ValueError("queen column is outside the board")
    if any(
        placed[i] == placed[j] or abs(placed[i] - placed[j]) == j - i
        for i in range(row)
        for j in range(i + 1, row)
    ):
        raise ValueError("starting queens attack one another")
    if row == n:
        return [placed]

    def safe(column: int) -> bool:
        return all(
            column != previous and abs(column - previous) != row - previous_row
            for previous_row, previous in enumerate(placed)
        )

    return [
        solution
        for column in range(n)
        if safe(column)
        for solution in queens(n, row + 1, placed + (column,))
    ]


def fib_iter(n: int) -> int:
    """Return Fibonacci using n iterations and two integer accumulators."""
    natural(n)
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def count_calls(n: int) -> int:
    """Count naive fib calls by the same recursion tree as the lecture."""
    natural(n)
    return 1 if n < 2 else 1 + count_calls(n - 1) + count_calls(n - 2)
