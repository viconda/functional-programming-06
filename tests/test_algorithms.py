"""Independent correctness and immutability checks for tasks 1, 2 and 4."""

from __future__ import annotations

import copy
import json
import math
import random
import unittest
from pathlib import Path

from lab06.backtracking import (
    combinations,
    maze_path,
    permutations,
    queens_count,
    subset_sum,
)
from lab06.dc import (
    Subarray,
    count_inversions,
    kadane_reduce,
    max_subarray_dc,
    merge_sort,
    naive_inversions,
    quick_sort,
)
from lab06.nested import (
    deep_depth,
    deep_flatten,
    deep_map,
    deep_sum,
    get_path,
    paths,
    set_path,
)


class DivideConquerTests(unittest.TestCase):
    """Check boundary cases and compare independent implementations."""

    def test_sorts_random_and_unchanged(self) -> None:
        """Both sorts match Python on 100 deterministic random inputs."""
        rng = random.Random(606)
        for _ in range(100):
            xs = [rng.randint(-15, 15) for _ in range(rng.randrange(65))]
            before = xs.copy()
            self.assertEqual(merge_sort(xs), sorted(xs))
            self.assertEqual(quick_sort(xs), sorted(xs))
            self.assertEqual(xs, before)

    def test_sort_boundaries(self) -> None:
        """Empty, duplicate, ascending, descending and text inputs work."""
        for xs in (
            [],
            [4],
            [3] * 200,
            list(range(100)),
            list(range(100, 0, -1)),
            ["я", "а", "б", "а"],
        ):
            self.assertEqual(merge_sort(xs), sorted(xs))
            self.assertEqual(quick_sort(xs), sorted(xs))
            self.assertIsNot(merge_sort(xs), xs)

    def test_inversions_100_random(self) -> None:
        """The O(n log n) counter agrees with the O(n²) definition."""
        rng = random.Random(610)
        for _ in range(100):
            xs = [rng.randint(-30, 30) for _ in range(rng.randrange(81))]
            before = xs.copy()
            self.assertEqual(count_inversions(xs), naive_inversions(xs))
            self.assertEqual(xs, before)

    def test_inversions_boundaries(self) -> None:
        """Equal values are not inversions; descending lists have n(n-1)/2."""
        self.assertEqual(count_inversions([]), 0)
        self.assertEqual(count_inversions([2, 2, 2]), 0)
        self.assertEqual(count_inversions(list(range(10, 0, -1))), 45)

    def test_subarray_against_exhaustive(self) -> None:
        """Both algorithms match all-interval search, including equal sums."""
        rng = random.Random(611)
        for _ in range(120):
            xs = [rng.randint(-6, 6) for _ in range(rng.randint(1, 20))]
            before = xs.copy()
            candidates = [
                Subarray(sum(xs[i:j]), i, j)
                for i in range(len(xs))
                for j in range(i + 1, len(xs) + 1)
            ]
            expected = max(
                candidates, key=lambda item: (item.total, -item.start, -item.stop)
            )
            self.assertEqual(max_subarray_dc(xs), expected)
            self.assertEqual(kadane_reduce(xs), expected)
            self.assertEqual(xs, before)

    def test_subarray_empty_and_negative(self) -> None:
        """Empty inputs fail explicitly and all-negative inputs stay nonempty."""
        for function in (max_subarray_dc, kadane_reduce):
            with self.assertRaises(ValueError):
                function([])
            self.assertEqual(function([-8, -2, -5]), Subarray(-2, 1, 2))
            self.assertEqual(function([0, 0]), Subarray(0, 0, 1))


class NestedTests(unittest.TestCase):
    """Check recursive JSON operations and container preservation."""

    def test_flatten_depth_sum(self) -> None:
        """Atoms, empty containers, bools and heterogeneous structures work."""
        value = [1, (2, {"x": [3.5, "text", True, None]}), []]
        before = copy.deepcopy(value)
        self.assertEqual(list(deep_flatten(value)), [1, 2, 3.5, "text", True, None])
        self.assertEqual(deep_depth(value), 4)
        self.assertEqual(deep_sum(value), 6.5)
        self.assertEqual(deep_depth(7), 0)
        self.assertEqual(deep_depth({}), 1)
        self.assertEqual(deep_sum([]), 0)
        self.assertEqual(value, before)

    def test_map_preserves_types(self) -> None:
        """Mapping values preserves keys, list/tuple kinds and input data."""
        value = [1, (2, {"key": 3})]
        before = copy.deepcopy(value)
        self.assertEqual(deep_map(lambda x: x * 10, value), [10, (20, {"key": 30})])
        self.assertEqual(value, before)

    def test_json_paths_and_copy(self) -> None:
        """Every generated leaf path can be read and set without mutation."""
        source = Path(__file__).resolve().parents[1] / "data" / "nested.json"
        doc = json.loads(source.read_text(encoding="utf-8"))
        before = copy.deepcopy(doc)
        self.assertGreaterEqual(deep_depth(doc), 4)
        for path, value in paths(doc):
            self.assertEqual(get_path(doc, path), value)
            changed = set_path(doc, path, "changed")
            self.assertEqual(get_path(changed, path), "changed")
            self.assertEqual(doc, before)
        self.assertEqual(get_path(doc, ()), doc)
        self.assertEqual(set_path(doc, (), 17), 17)

    def test_paths_empty_and_tuple(self) -> None:
        """Empty collections are leaves and tuple replacement stays tuple."""
        self.assertEqual(list(paths({"a": []})), [(("a",), [])])
        self.assertEqual(set_path((1, (2,)), (1, 0), 9), (1, (9,)))

    def test_path_errors(self) -> None:
        """Missing keys, bad indices and attempts to traverse atoms fail."""
        for function in (lambda d, p: get_path(d, p), lambda d, p: set_path(d, p, 9)):
            with self.assertRaises(KeyError):
                function({}, ("missing",))
            with self.assertRaises(IndexError):
                function([1], (1,))
            with self.assertRaises(IndexError):
                function([1], (-1,))
            with self.assertRaises(TypeError):
                function([1], ("0",))
            with self.assertRaises(TypeError):
                function(1, (0,))


class BacktrackingTests(unittest.TestCase):
    """Check generation counts, signed subset sums, queens and mazes."""

    def test_permutations(self) -> None:
        """Positional permutations have factorial size and expected contents."""
        xs = [1, 2, 3, 4]
        result = list(permutations(xs))
        self.assertEqual(len(result), math.factorial(len(xs)))
        self.assertEqual(len(set(result)), 24)
        self.assertTrue(all(sorted(item) == xs for item in result))
        self.assertEqual(xs, [1, 2, 3, 4])
        self.assertEqual(list(permutations([])), [()])

    def test_combinations(self) -> None:
        """All sizes agree with the binomial formula, including k=0 and k>n."""
        for n in range(8):
            for k in range(n + 2):
                result = list(combinations(list(range(n)), k))
                self.assertEqual(len(result), math.comb(n, k) if k <= n else 0)
                self.assertTrue(all(tuple(sorted(item)) == item for item in result))
        with self.assertRaises(ValueError):
            list(combinations([1, 2], -1))

    def test_subset_sum_random_exhaustive(self) -> None:
        """Signed subsets match an independent binary-mask enumeration."""
        rng = random.Random(612)
        for _ in range(60):
            xs = [rng.randint(-3, 6) for _ in range(rng.randrange(9))]
            before = xs.copy()
            target = rng.randint(-7, 15)
            expected = []
            for mask in range(1 << len(xs)):
                values = tuple(x for i, x in enumerate(xs) if mask & (1 << i))
                if sum(values) == target:
                    expected.append(values)
            self.assertEqual(sorted(subset_sum(xs, target)), sorted(expected))
            self.assertEqual(xs, before)

    def test_subset_sum_special_cases(self) -> None:
        """Do not cut signed sums early or omit zero extensions of solutions."""
        self.assertEqual(list(subset_sum([8, -3], 5)), [(8, -3)])
        self.assertEqual(list(subset_sum([0, 0], 0)), [(), (0,), (0,), (0, 0)])
        self.assertEqual(list(subset_sum([], 0)), [()])
        self.assertEqual(list(subset_sum([1, 2], -1)), [])

    def test_queens_known_counts(self) -> None:
        """Early pruning and leaf-only checks produce the same known counts."""
        for n, expected in (
            (0, 1),
            (1, 1),
            (2, 0),
            (3, 0),
            (4, 2),
            (5, 10),
            (6, 4),
            (7, 40),
            (8, 92),
        ):
            self.assertEqual(queens_count(n), expected)
            self.assertEqual(queens_count(n, prune=False), expected)
        with self.assertRaises(ValueError):
            queens_count(-1)

    def test_maze_valid_path(self) -> None:
        """A returned path joins S and E with adjacent passable unique cells."""
        maze = (
            "#########",
            "#S..#...#",
            "#.#.#.#E#",
            "#.#...#.#",
            "#.#####.#",
            "#.......#",
            "#########",
        )
        before = tuple(maze)
        path = maze_path(maze)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path[0], (1, 1))
        self.assertEqual(path[-1], (2, 7))
        self.assertEqual(len(set(path)), len(path))
        for (r1, c1), (r2, c2) in zip(path, path[1:]):
            self.assertEqual(abs(r1 - r2) + abs(c1 - c2), 1)
            self.assertNotEqual(maze[r2][c2], "#")
        self.assertEqual(maze, before)

    def test_maze_unreachable_and_invalid(self) -> None:
        """Closed exits return None; malformed mazes fail explicitly."""
        self.assertIsNone(maze_path(("#####", "#S#E#", "#####")))
        for maze in ((), ("S", "..E"), ("S..",), ("SxE",), ("SSE",)):
            with self.assertRaises(ValueError):
                maze_path(maze)


if __name__ == "__main__":
    unittest.main(verbosity=2)
