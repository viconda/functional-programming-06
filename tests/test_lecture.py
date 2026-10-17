"""Independent behavior checks for every lecture example and all four tasks."""

from __future__ import annotations

import contextlib
import copy
import io
import itertools
import math
import random
import sys
import unittest
from functools import reduce
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lecture06"))
import database as db
import examples as ex


class ExampleTests(unittest.TestCase):
    def test_factorial_against_standard_library(self) -> None:
        for n in range(30):
            self.assertEqual(ex.factorial(n), math.factorial(n))
            self.assertEqual(ex.factorial_acc(n), math.factorial(n))

    def test_invalid_natural_arguments(self) -> None:
        for function in (
            ex.factorial,
            ex.fib,
            ex.is_even,
            ex.is_odd,
            ex.factorial_acc,
            ex.fib_iter,
            ex.count_calls,
            ex.depth,
        ):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function(-1)
                with self.assertRaises(TypeError):
                    function(1.5)
                with self.assertRaises(TypeError):
                    function(True)

    def test_trace_contains_calls_and_returns(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(ex.factorial_traced(3), 6)
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 8)
        self.assertEqual(lines[3], "      вызов factorial(0)")
        self.assertEqual(lines[-1], "возврат 6")

    def test_recursion_limit_is_observed_without_changing_it(self) -> None:
        limit = sys.getrecursionlimit()
        with self.assertRaises(RecursionError):
            ex.depth(limit + 100)
        self.assertEqual(sys.getrecursionlimit(), limit)
        self.assertEqual(ex.depth(20), 20)

    def test_mutual_and_nested_recursion(self) -> None:
        for n in range(30):
            self.assertEqual(ex.is_even(n), n % 2 == 0)
            self.assertEqual(ex.is_odd(n), n % 2 == 1)
        self.assertEqual(ex.ackermann(2, 3), 9)
        self.assertEqual(ex.ackermann(0, 0), 1)
        self.assertEqual(ex.ackermann(1, 3), 5)
        with self.assertRaises(ValueError):
            ex.ackermann(-1, 1)

    def test_fast_pow_against_builtin(self) -> None:
        for x in range(-4, 5):
            for n in range(21):
                self.assertEqual(ex.fast_pow(x, n), x**n)
        with self.assertRaises(ValueError):
            ex.fast_pow(2, -1)

    def test_gcd_against_standard_library(self) -> None:
        for a in range(-15, 16):
            for b in range(-15, 16):
                self.assertEqual(ex.gcd(a, b), math.gcd(a, b))

    def test_hanoi_simulates_legal_moves(self) -> None:
        for n in range(9):
            pegs = {"A": list(range(n, 0, -1)), "B": [], "C": []}
            moves = ex.hanoi(n)
            self.assertEqual(len(moves), 2**n - 1)
            for disk, source, destination in moves:
                self.assertEqual(pegs[source].pop(), disk)
                self.assertTrue(not pegs[destination] or pegs[destination][-1] > disk)
                pegs[destination].append(disk)
            self.assertEqual(pegs["C"], list(range(n, 0, -1)))
            self.assertFalse(pegs["A"] or pegs["B"])

    def test_hanoi_rejects_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            ex.hanoi(-1)
        with self.assertRaises(ValueError):
            ex.hanoi(2, "A", "A", "B")

    def test_binary_search_against_membership(self) -> None:
        values = tuple(range(0, 100, 3))
        for target in range(-2, 102):
            expected = values.index(target) if target in values else -1
            self.assertEqual(ex.binary_search(values, target), expected)
        self.assertEqual(ex.binary_search((), 4), -1)
        self.assertIn(ex.binary_search((1, 2, 2, 3), 2), (1, 2))

    def test_sorts_against_builtin_and_preserve_inputs(self) -> None:
        generator = random.Random(606)
        cases = [[], [1], [2, 2, 2], list(range(40)), list(range(40, -1, -1))]
        cases += [[generator.randrange(-15, 16) for _ in range(40)] for _ in range(20)]
        for function in (ex.merge_sort, ex.quicksort):
            for values in cases:
                original = values.copy()
                self.assertEqual(function(values), sorted(values))
                self.assertEqual(values, original)

    def test_merge_and_no_alias_at_base_case(self) -> None:
        right = [1, 3]
        self.assertEqual(ex.merge([1, 4], [2, 3]), [1, 2, 3, 4])
        result = ex.merge([], right)
        result.append(5)
        self.assertEqual(right, [1, 3])

    def test_head_tail_operations(self) -> None:
        for values in ([], [1], [1, 2, 3, 4]):
            original = values.copy()
            self.assertEqual(ex.length(values), len(values))
            self.assertEqual(
                ex.my_map(lambda x: x * 10, values),
                list(map(lambda x: x * 10, values)),
            )
            self.assertEqual(
                ex.my_filter(lambda x: bool(x % 2), values),
                list(filter(lambda x: x % 2, values)),
            )
            self.assertEqual(ex.fold_left(lambda a, x: a - x, 0, values), -sum(values))
            self.assertEqual(values, original)

    def test_flatten_depth_and_input_immutability(self) -> None:
        values = [1, [2, [3, [4, 5]], 6], [[7]], 8]
        original = copy.deepcopy(values)
        self.assertEqual(ex.flatten(values), list(range(1, 9)))
        self.assertEqual(ex.nested_depth(values), 4)
        self.assertEqual(values, original)
        self.assertEqual(ex.flatten([[], (1, (2,))]), [1, 2])
        self.assertEqual([ex.nested_depth(x) for x in (7, [], [[]])], [0, 1, 2])

    def test_persistent_bst_versions_and_sharing(self) -> None:
        first = reduce(ex.insert, [50, 30, 70, 20, 40, 60, 80], None)
        second = ex.insert(first, 65)
        self.assertEqual(ex.inorder(first), [20, 30, 40, 50, 60, 70, 80])
        self.assertEqual(ex.inorder(second), [20, 30, 40, 50, 60, 65, 70, 80])
        self.assertIs(second.left, first.left)
        self.assertEqual(ex.insert(second, 65), second)
        self.assertEqual(ex.height(second), 4)
        self.assertFalse(ex.contains(first, 65))
        self.assertTrue(ex.contains(second, 65))
        self.assertEqual(ex.height(None), 0)

    def test_find_values_at_multiple_levels(self) -> None:
        document = {"x": 1, "children": [{"x": 2}, {"other": {"x": 3}}]}
        original = copy.deepcopy(document)
        self.assertEqual(ex.find_values(document, "x"), [1, 2, 3])
        self.assertEqual(ex.find_values(document, "missing"), [])
        self.assertEqual(ex.find_values(4, "x"), [])
        self.assertEqual(document, original)

    def test_permutations_against_itertools(self) -> None:
        for values in ([], [1], [1, 2, 3], [1, 1, 2]):
            self.assertEqual(
                ex.permutations(values),
                [list(p) for p in itertools.permutations(values)],
            )

    def test_subsets_against_combinations(self) -> None:
        for size in range(6):
            values = list(range(size))
            expected = {
                tuple(c)
                for k in range(size + 1)
                for c in itertools.combinations(values, k)
            }
            self.assertEqual({tuple(s) for s in ex.subsets(values)}, expected)
            self.assertEqual(len(ex.subsets(values)), 2**size)

    def test_all_queen_solutions_are_valid_and_unique(self) -> None:
        for n, count in ((0, 1), (1, 1), (2, 0), (3, 0), (4, 2), (6, 4), (8, 92)):
            solutions = ex.queens(n)
            self.assertEqual(len(solutions), count)
            self.assertEqual(len(set(solutions)), count)
            for solution in solutions:
                self.assertEqual(len(solution), n)
                self.assertEqual(len(set(solution)), n)
                self.assertTrue(
                    all(
                        abs(solution[i] - solution[j]) != j - i
                        for i in range(n)
                        for j in range(i + 1, n)
                    )
                )

    def test_fibonacci_counts_against_closed_identity(self) -> None:
        for n in range(18):
            self.assertEqual(ex.fib(n), ex.fib_iter(n))
            self.assertEqual(ex.count_calls(n), 2 * ex.fib_iter(n + 1) - 1)
        self.assertEqual(ex.fib_iter(30), 832040)
        self.assertEqual(
            [ex.count_calls(n) for n in (10, 20, 25)], [177, 21891, 242785]
        )


class DatabaseTests(unittest.TestCase):
    def test_select_project_and_order(self) -> None:
        self.assertEqual(
            [r.id for r in db.select(db.STUDENTS, lambda r: r.age > 18)], [1, 3, 4]
        )
        projected = db.project(db.STUDENTS, ("name", "age"))
        self.assertEqual(projected[0]._fields, ("name", "age"))
        self.assertEqual(projected[0].name, "Швачкина Виктория")
        self.assertEqual([r.id for r in db.order_by(db.STUDENTS, "age")], [2, 1, 3, 4])
        self.assertEqual(db.project((), ("name",)), ())

    def test_insert_update_delete_preserve_old_versions(self) -> None:
        original = db.STUDENTS
        inserted = db.insert(original, db.Student(5, "Елена", 19, "Шымкент"))
        updated = db.update(original, lambda r: r.id == 2, {"age": 19})
        deleted = db.delete(original, lambda r: r.id == 3)
        self.assertEqual(len(inserted), 5)
        self.assertEqual(original[1].age, 18)
        self.assertEqual(updated[1].age, 19)
        self.assertEqual([r.id for r in deleted], [1, 2, 4])
        self.assertEqual(len(original), 4)
        self.assertIs(updated[0], original[0])

    def test_separate_projections_share_a_structural_schema(self) -> None:
        first = db.project(db.STUDENTS[:1], ("name", "age"))
        second = db.project(db.STUDENTS[1:2], ("name", "age"))
        self.assertIsNot(type(first[0]), type(second[0]))
        combined = db.insert(first, second[0])
        self.assertEqual(combined, first + second)
        self.assertEqual(len(first), 1)
        self.assertEqual(
            [row.name for row in db.order_by(combined, "age")],
            ["Айдана", "Швачкина Виктория"],
        )
        updated = db.update(combined, lambda row: row.age == 18, {"age": 19})
        self.assertEqual([row.age for row in updated], [19, 19])
        self.assertEqual(second[0].age, 18)

    def test_projection_field_order_is_part_of_schema(self) -> None:
        names_first = db.project(db.STUDENTS[:1], ("name", "age"))
        ages_first = db.project(db.STUDENTS[:1], ("age", "name"))
        with self.assertRaises(TypeError):
            db.insert(names_first, ages_first[0])
        with self.assertRaises(TypeError):
            db.select(names_first + ages_first, lambda _: True)

    def test_empty_relational_operations(self) -> None:
        self.assertEqual(db.select((), lambda _: True), ())
        self.assertEqual(db.order_by((), "age"), ())
        self.assertEqual(db.update((), lambda _: True, {"age": 20}), ())
        self.assertEqual(db.delete((), lambda _: True), ())
        self.assertEqual(db.join((), db.GRADES, "id", "student_id"), ())
        self.assertEqual(db.group_by((), "city", len), ())
        self.assertEqual(db.insert((), db.STUDENTS[0]), (db.STUDENTS[0],))

    def test_invalid_schema_and_mutable_cells(self) -> None:
        with self.assertRaises(ValueError):
            db.project(db.STUDENTS, ("unknown",))
        with self.assertRaises(ValueError):
            db.project(db.STUDENTS, ("age", "age"))
        with self.assertRaises(ValueError):
            db.update(db.STUDENTS, lambda _: True, {"missing": 1})
        with self.assertRaises(TypeError):
            db.insert(db.STUDENTS, db.GRADES[0])
        with self.assertRaises(TypeError):
            db.update(db.STUDENTS, lambda _: True, {"name": []})

        class BadRow(NamedTuple):
            value: list

        with self.assertRaises(TypeError):
            db.insert((), BadRow([]))

    def test_join_preserves_many_to_many_matches(self) -> None:
        joined = db.join(db.STUDENTS, db.GRADES, "id", "student_id")
        self.assertEqual(len(joined), 6)
        repeated = db.join((db.STUDENTS[0],) * 2, db.GRADES[:2], "id", "student_id")
        self.assertEqual(len(repeated), 4)
        self.assertEqual(
            db.join(db.STUDENTS, (db.Grade(99, "x", 1),), "id", "student_id"), ()
        )
        self.assertIs(joined[0].left, db.STUDENTS[0])

    def test_group_by_averages_counts_and_order(self) -> None:
        grouped = db.group_by(
            db.GRADES,
            "student_id",
            lambda rows: sum(r.score for r in rows) / len(rows),
        )
        self.assertEqual(
            [(g.key, g.value) for g in grouped],
            [(1, 92.5), (2, 88.0), (3, 76.0), (4, 94.0)],
        )
        cities = db.group_by(db.STUDENTS, "city", len)
        self.assertEqual(cities, (db.Group("Алматы", 3), db.Group("Астана", 1)))
        with self.assertRaises(TypeError):
            db.group_by(db.STUDENTS, "city", list)

    def test_group_by_nan_keys_keep_nonempty_groups(self) -> None:
        class Measurement(NamedTuple):
            key: float
            value: int

        shared_nan = float("nan")
        separate_nan = float("nan")
        table = (
            Measurement(shared_nan, 10),
            Measurement(1.0, 8),
            Measurement(shared_nan, 20),
            Measurement(separate_nan, 30),
        )
        grouped = db.group_by(table, "key", lambda rows: tuple(r.value for r in rows))
        self.assertEqual([group.value for group in grouped], [(10, 20), (8,), (30,)])
        self.assertIs(grouped[0].key, shared_nan)
        self.assertIs(grouped[2].key, separate_nan)
        averaged = db.group_by(
            table, "key", lambda rows: sum(row.value for row in rows) / len(rows)
        )
        self.assertEqual([group.value for group in averaged], [15.0, 8.0, 30.0])

    def test_group_by_fresh_nan_callable_preserves_every_row(self) -> None:
        grouped = db.group_by(db.STUDENTS, lambda _: float("nan"), len)
        self.assertEqual(len(grouped), len(db.STUDENTS))
        self.assertTrue(all(math.isnan(group.key) for group in grouped))
        self.assertEqual([group.value for group in grouped], [1, 1, 1, 1])

    def test_predicate_compilation_and_nested_boolean_operators(self) -> None:
        expression = ("and", ("gt", "age", 18), ("eq", "city", "Алматы"))
        compiled = db.compile_predicate(expression)
        self.assertEqual(
            [r.id for r in db.select(db.STUDENTS, compiled)],
            [1, 4],
        )
        nested = ("or", ("not", ("eq", "city", "Алматы")), ("lt", "age", 19))
        compiled_nested = db.compile_predicate(nested)
        self.assertEqual(
            [r.id for r in db.select(db.STUDENTS, compiled_nested)],
            [2, 3],
        )
        for op, expected in (
            ("eq", True),
            ("ne", False),
            ("gt", False),
            ("ge", True),
            ("lt", False),
            ("le", True),
        ):
            self.assertEqual(
                db.compile_predicate((op, "age", 19))(db.STUDENTS[0]), expected
            )

    def test_predicate_rejects_malformed_ast(self) -> None:
        cases = (
            (),
            [],
            ("and",),
            ("and", ("eq", "age", 19)),
            ("not",),
            ("eq", "age"),
            ("eq", 1, 19),
            ("unknown", "age", 19),
            ([], "age", 19),
        )
        for expression in cases:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    db.compile_predicate(expression)
        with self.assertRaises(AttributeError):
            db.compile_predicate(("eq", "absent", 1))(db.STUDENTS[0])

    def test_boolean_short_circuit(self) -> None:
        false_and_missing = ("and", ("eq", "age", -1), ("eq", "absent", 1))
        true_or_missing = ("or", ("eq", "age", 19), ("eq", "absent", 1))
        self.assertFalse(db.compile_predicate(false_and_missing)(db.STUDENTS[0]))
        self.assertTrue(db.compile_predicate(true_or_missing)(db.STUDENTS[0]))

    def test_integrated_student_grade_query(self) -> None:
        predicate = db.compile_predicate(
            ("and", ("gt", "age", 18), ("eq", "city", "Алматы"))
        )
        joined = db.join(
            db.select(db.STUDENTS, predicate), db.GRADES, "id", "student_id"
        )
        grouped = db.group_by(
            joined,
            lambda r: r.left.name,
            lambda rows: sum(r.right.score for r in rows) / len(rows),
        )
        result = db.order_by(grouped, lambda r: -r.value)
        self.assertEqual(
            result, (db.Group("Мария", 94.0), db.Group("Швачкина Виктория", 92.5))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
