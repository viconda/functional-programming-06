"""Run each lecture example and independent assignment as real Python code."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from functools import reduce
from pathlib import Path
from typing import Callable

import database as db
import examples as ex


def e01() -> None:
    """Factorials from zero to seven and the invalid-input check."""
    print([ex.factorial(i) for i in range(8)])
    assert ex.factorial(0) == 1 and ex.factorial(7) == 5040
    try:
        ex.factorial(-1)
    except ValueError as error:
        print("factorial(-1):", type(error).__name__, str(error))


def e02() -> None:
    """Print the actual recursive call stack trace for factorial(3)."""
    assert ex.factorial_traced(3) == 6


def e03() -> None:
    """Observe the interpreter limit without increasing it."""
    print("sys.getrecursionlimit() =", sys.getrecursionlimit())
    try:
        ex.depth(10_000)
    except RecursionError as error:
        print("RecursionError:", error)
    else:
        raise AssertionError("expected the default limit to reject depth(10000)")
    print("depth(10) =", ex.depth(10))


def e04() -> None:
    """Run every recursion kind shown in the lecture's combined block."""
    print("fib(10) =", ex.fib(10))
    print("is_even(10) =", ex.is_even(10))
    print("is_odd(7) =", ex.is_odd(7))
    print("factorial_acc(10) =", ex.factorial_acc(10))
    print("ackermann(2, 3) =", ex.ackermann(2, 3))
    assert (ex.fib(10), ex.factorial_acc(10), ex.ackermann(2, 3)) == (55, 3628800, 9)
    assert ex.is_even(0) and not ex.is_odd(0)
    print("Нулевые входы взаимной рекурсии: OK")


def e05() -> None:
    """Validate powers against Python's operator."""
    print("fast_pow(2, 100) == 2 ** 100:", ex.fast_pow(2, 100) == 2**100)
    print("fast_pow(3, 13) =", ex.fast_pow(3, 13))
    print("fast_pow(7, 0) =", ex.fast_pow(7, 0))
    assert ex.fast_pow(3, 13) == 1594323 and ex.fast_pow(7, 0) == 1


def e06() -> None:
    """Check Euclid's algorithm, signs and zero arguments."""
    for a, b in ((1071, 462), (17, 5), (-18, 24), (0, 0), (0, 9)):
        print(f"gcd({a}, {b}) = {ex.gcd(a, b)}")
    assert ex.gcd(1071, 462) == 21 and ex.gcd(-18, 24) == 6


def e07() -> None:
    """Generate and simulate legal Hanoi moves."""
    moves = ex.hanoi(3)
    print("hanoi(3), ходов:", len(moves))
    for number, move in enumerate(moves, 1):
        print(number, move)
    print("len(hanoi(10)) =", len(ex.hanoi(10)))
    pegs = {"A": [3, 2, 1], "B": [], "C": []}
    for disk, source, destination in moves:
        assert pegs[source].pop() == disk
        assert not pegs[destination] or pegs[destination][-1] > disk
        pegs[destination].append(disk)
    assert pegs == {"A": [], "B": [], "C": [3, 2, 1]}
    print("Симуляция правил: OK; итог:", pegs)


def e08() -> None:
    """Search values present in, absent from, and outside the sequence."""
    data = tuple(range(0, 100, 3))
    print("data = tuple(range(0, 100, 3))")
    for target in (42, 43, 0, 99, 100):
        print(f"binary_search(data, {target}) =", ex.binary_search(data, target))
    print("binary_search((), 1) =", ex.binary_search((), 1))
    assert ex.binary_search(data, 42) == 14 and ex.binary_search(data, 43) == -1


def e09() -> None:
    """Run merge sort and confirm that the input list stays unchanged."""
    values = [38, 27, 43, 3, 9, 82, 10]
    original = values.copy()
    print("Вход:", values)
    print("merge_sort:", ex.merge_sort(values))
    print("Пустой список:", ex.merge_sort([]))
    print("Повторы:", ex.merge_sort([2, 1, 2, -1]))
    assert ex.merge_sort(values) == sorted(values) and values == original
    print("Исходный список сохранён:", values == original)


def e10() -> None:
    """Run functional quicksort, including duplicates and sorted data."""
    for values in ([5, 3, 8, 1, 9, 2, 7], [], [2, 2, 1], [1, 2, 3]):
        original = values.copy()
        result = ex.quicksort(values)
        print(values, "->", result)
        assert result == sorted(values) and values == original
    print("Сортировка и неизменность всех входов: OK")


def e11() -> None:
    """Run each of the four head/tail operations."""
    numbers = [1, 2, 3, 4, 5, 6]
    print("nums =", numbers)
    print("length =", ex.length(numbers))
    print("my_map(x * 10) =", ex.my_map(lambda value: value * 10, numbers))
    print("my_filter(x % 2) =", ex.my_filter(lambda value: bool(value % 2), numbers))
    print("fold_left(sum) =", ex.fold_left(lambda acc, value: acc + value, 0, numbers))
    print(
        "fold_left(subtract, 0, [1,2,3]) =",
        ex.fold_left(lambda a, x: a - x, 0, [1, 2, 3]),
    )
    assert ex.length([]) == 0 and ex.fold_left(lambda a, x: a + x, 7, []) == 7
    print("Пустые входы: length=0, fold_left возвращает исходный аккумулятор")


def e12() -> None:
    """Flatten nested lists and inspect the chosen depth convention."""
    data = [1, [2, [3, [4, 5]], 6], [[7]], 8]
    print("Вход:", data)
    print("flatten =", ex.flatten(data))
    print("nested_depth =", ex.nested_depth(data))
    print("flatten([[], (1, (2,))]) =", ex.flatten([[], (1, (2,))]))
    print("Глубины числа 7, [], [[]]:", [ex.nested_depth(x) for x in (7, [], [[]])])
    assert ex.flatten(data) == list(range(1, 9)) and ex.nested_depth(data) == 4


def e13() -> None:
    """Build two persistent BST versions and prove subtree sharing."""
    first = reduce(ex.insert, [50, 30, 70, 20, 40, 60, 80], None)
    second = ex.insert(first, 65)
    print("inorder(t1) =", ex.inorder(first))
    print("inorder(t2) =", ex.inorder(second))
    print("height(t2) =", ex.height(second))
    print(
        "contains(t2,65), contains(t1,65) =",
        ex.contains(second, 65),
        ex.contains(first, 65),
    )
    print("t2.left is t1.left:", second.left is first.left)
    print("Повторная вставка 65 сохраняет содержимое:", ex.insert(second, 65) == second)
    assert second.left is first.left and not ex.contains(first, 65)


def e14() -> None:
    """Search every nesting level of a JSON-like university document."""
    university = {
        "name": "Satbayev University",
        "institutes": [
            {
                "name": "ИАиИТ",
                "departments": [
                    {"name": "Программная инженерия", "staff": 40},
                    {"name": "Кибербезопасность", "staff": 25},
                ],
            },
            {
                "name": "ИГиНГД",
                "departments": [
                    {"name": "Геология", "staff": 30},
                ],
            },
        ],
    }
    values = ex.find_values(university, "staff")
    print("find_values(university, 'staff') =", values)
    print("sum =", sum(values))
    print("Неизвестный ключ:", ex.find_values(university, "absent"))
    print("Скалярный вход:", ex.find_values(42, "staff"))
    assert values == [40, 25, 30] and sum(values) == 95


def e15() -> None:
    """Execute all three backtracking demonstrations."""
    print("permutations([1, 2, 3]):")
    for permutation in ex.permutations([1, 2, 3]):
        print(permutation)
    print("subsets(['a', 'b', 'c']):")
    print(ex.subsets(["a", "b", "c"]))
    six, eight = ex.queens(6), ex.queens(8)
    print("queens(6):", len(six), "решения; первое:", six[0])
    print("queens(8):", len(eight), "решения")
    print("queens(0), queens(2), queens(3):", ex.queens(0), ex.queens(2), ex.queens(3))
    assert len(six) == 4 and len(eight) == 92


def e16() -> None:
    """Compare iterative Fibonacci with actual recursive call counts."""
    print("fib_iter(30) =", ex.fib_iter(30))
    for n in (10, 20, 25):
        print(f"count_calls({n}) = {ex.count_calls(n)}")
    assert all(ex.fib(n) == ex.fib_iter(n) for n in range(16))
    print("Сравнение fib и fib_iter на n=0..15: OK")


def s01() -> None:
    """Demonstrate all six pure relational operations separately."""
    original = db.STUDENTS
    print("Исходная таблица:", len(original), "записи; tuple + NamedTuple")
    selected = db.select(original, lambda row: row.age > 18)
    print("select(age > 18):", [row.id for row in selected])
    projected = db.project(original, ("name", "age"))
    print("project(name, age):", tuple(tuple(row) for row in projected))
    print("order_by(age):", [row.id for row in db.order_by(original, "age")])
    added = db.insert(original, db.Student(5, "Елена", 19, "Шымкент"))
    print("insert: было", len(original), "; стало", len(added))
    updated = db.update(original, lambda row: row.id == 2, {"age": 19})
    print("update(id=2): age", original[1].age, "->", updated[1].age)
    deleted = db.delete(original, lambda row: row.id == 3)
    print("delete(id=3):", [row.id for row in deleted])
    assert len(original) == 4 and original[1].age == 18 and len(added) == 5
    print("Исходная версия после всех операций:", [row.id for row in original])
    print("Проверка неизменности: OK")


def s02() -> None:
    """Demonstrate joins, aggregates and no-match behavior."""
    joined = db.join(db.STUDENTS, db.GRADES, "id", "student_id")
    print("join(Students, Grades):", len(joined), "строк")
    for row in joined:
        print(f"{row.left.name}: {row.right.course} = {row.right.score}")
    grouped = db.group_by(
        db.GRADES,
        "student_id",
        lambda rows: sum(r.score for r in rows) / len(rows),
    )
    print("group_by(student_id, среднее):", [(g.key, g.value) for g in grouped])
    cities = db.group_by(db.STUDENTS, "city", len)
    print("group_by(city, len):", [(g.key, g.value) for g in cities])
    print("join с пустой таблицей:", db.join(db.STUDENTS, (), "id", "student_id"))
    assert len(joined) == 6 and grouped[0].value == 92.5


def s03() -> None:
    """Compile nested conditions and report invalid syntax explicitly."""
    expression = ("and", ("gt", "age", 18), ("eq", "city", "Алматы"))
    print("Условие:", expression)
    result = db.select(db.STUDENTS, db.compile_predicate(expression))
    print("Выбраны:", [row.name for row in result])
    nested = ("or", ("not", ("eq", "city", "Алматы")), ("lt", "age", 19))
    nested_rows = db.select(db.STUDENTS, db.compile_predicate(nested))
    print("Вложенное or/not:", [row.id for row in nested_rows])
    try:
        db.compile_predicate(("unknown", "age", 18))
    except ValueError as error:
        print("Ошибочный оператор:", type(error).__name__, str(error))
    assert [row.id for row in result] == [1, 4]


def s04() -> None:
    """Run an integrated Students/Grades query and preserve data versions."""
    condition = ("and", ("gt", "age", 18), ("eq", "city", "Алматы"))
    adults = db.select(db.STUDENTS, db.compile_predicate(condition))
    joined = db.join(adults, db.GRADES, "id", "student_id")
    grouped = db.group_by(
        joined,
        lambda row: row.left.name,
        lambda rows: sum(row.right.score for row in rows) / len(rows),
    )
    rating = db.order_by(grouped, lambda group: (-group.value, group.key))
    print("Студенты старше 18 лет из Алматы, средний балл по убыванию:")
    for group in rating:
        print(f"{group.key}: {group.value:.2f}")
    changed = db.update(
        db.GRADES,
        lambda row: row.student_id == 1 and row.course == "Алгоритмы",
        {"score": 94},
    )
    print("Коррекция оценки, исходная версия:", db.GRADES[1].score)
    print("Новая версия:", changed[1].score)
    print("Студентов:", len(db.STUDENTS), "; оценок:", len(db.GRADES))
    print("Старые таблицы доступны; побочных изменений нет")
    assert [(row.key, row.value) for row in rating] == [
        ("Мария", 94.0),
        ("Швачкина Виктория", 92.5),
    ]
    assert db.GRADES[1].score == 90 and changed[1].score == 94


SECTIONS: dict[str, tuple[str, Callable[[], None]]] = {
    "E01": ("Факториал", e01),
    "E02": ("Трассировка стека факториала", e02),
    "E03": ("Ограничение глубины рекурсии", e03),
    "E04": ("Виды рекурсии", e04),
    "E05": ("Быстрое возведение в степень", e05),
    "E06": ("Алгоритм Евклида", e06),
    "E07": ("Ханойские башни", e07),
    "E08": ("Бинарный поиск", e08),
    "E09": ("Сортировка слиянием", e09),
    "E10": ("Быстрая сортировка", e10),
    "E11": ("Обработка списка: голова и хвост", e11),
    "E12": ("Вложенные списки", e12),
    "E13": ("Неизменяемое двоичное дерево", e13),
    "E14": ("Обход JSON-документа", e14),
    "E15": ("Перебор с возвратом", e15),
    "E16": ("Сравнение рекурсии и итерации", e16),
    "S01": ("СРС 1. Чистые операции над таблицей", s01),
    "S02": ("СРС 2. Соединение и группировка", s02),
    "S03": ("СРС 3. Рекурсивный разбор условия", s03),
    "S04": ("СРС 4. Демонстрация функциональной СУБД", s04),
}


def main() -> None:
    """Run one section or every section, optionally exporting actual output."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=tuple(SECTIONS), help="Run one section")
    parser.add_argument("--export", type=Path, help="Save actual outputs as JSON")
    arguments = parser.parse_args()
    selected = [arguments.section] if arguments.section else list(SECTIONS)
    evidence = []
    for identifier in selected:
        title, function = SECTIONS[identifier]
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            function()
            print("Проверки секции: OK")
        output = captured.getvalue()
        print(f"\n{identifier}. {title}\n{output}", end="")
        evidence.append(
            {
                "id": identifier,
                "title": title,
                "command": f"python lecture06/demo_lecture.py --section {identifier}",
                "output": output,
            }
        )
    if arguments.export:
        arguments.export.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
