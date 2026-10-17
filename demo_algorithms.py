"""Run task 1, 2 and 4 demonstrations and record real output as JSON."""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import platform
import random
import statistics
import time
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from lab06.backtracking import (
    combinations,
    maze_path,
    permutations,
    queens_count,
    subset_sum,
)
from lab06.dc import (
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


def load_document() -> Any:
    """Load the exercise JSON file using the standard json module."""
    source = Path(__file__).resolve().parent / "data" / "nested.json"
    return json.loads(source.read_text(encoding="utf-8"))


def timed(function: Callable[[], Any], repeats: int = 3) -> tuple[Any, float]:
    """Return the last result and median execution time over actual calls."""
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
    return result, statistics.median(samples)


def demo_l1a() -> dict[str, Any]:
    """Demonstrate both sorting algorithms and preservation of input."""
    xs = [8, 3, 5, 3, -2, 9, 0, 8]
    before = xs.copy()
    print("Исходный список:", xs)
    print(
        "Медиана первого, среднего и последнего:",
        sorted((xs[0], xs[len(xs) // 2], xs[-1]))[1],
    )
    print("Сортировка слиянием:", merge_sort(xs))
    print("Быстрая сортировка:", quick_sort(xs))
    print("Проверка sorted:", merge_sort(xs) == quick_sort(xs) == sorted(xs))
    print("Пустой список:", merge_sort([]), quick_sort([]))
    print("Одинаковые элементы:", quick_sort([5, 5, 5]))
    print("Вход не изменился:", xs == before)
    assert xs == before and merge_sort(xs) == quick_sort(xs) == sorted(xs)
    return {"input": xs, "sorted": sorted(xs)}


def demo_l1b() -> dict[str, Any]:
    """Compare inversion counting with the definition on 100 random lists."""
    example = [2, 4, 1, 3, 5]
    print("Пример:", example)
    print(
        "Число инверсий: быстрый =",
        count_inversions(example),
        "; наивный =",
        naive_inversions(example),
    )
    rng = random.Random(610)
    comparisons = 0
    for _ in range(100):
        xs = [rng.randint(-30, 30) for _ in range(rng.randrange(81))]
        before = xs.copy()
        assert count_inversions(xs) == naive_inversions(xs)
        assert xs == before
        comparisons += 1
    print("Seed = 610; длины 0..80; значения -30..30")
    print("Совпадений на случайных списках:", comparisons, "/ 100")
    print("Пустой список:", count_inversions([]))
    print("Равные элементы [2, 2, 2]:", count_inversions([2, 2, 2]))
    print("Убывающий список длины 10:", count_inversions(list(range(10, 0, -1))))
    print("Неизменность всех входных списков: PASS")
    return {"random_comparisons": comparisons, "example_inversions": 3}


def demo_l1c() -> dict[str, Any]:
    """Compare maximum subarrays, negative cases and measured runtimes."""
    xs = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    result = max_subarray_dc(xs)
    other = kadane_reduce(xs)
    print("Исходный список:", xs)
    print("Разделяй и властвуй:", result)
    print("Кадане через reduce:", other)
    print("Подмассив:", xs[result.start : result.stop])
    print("Индексы с нуля, правая граница исключена.")
    print("Все отрицательные:", max_subarray_dc([-8, -2, -5]))
    assert result == other
    rng = random.Random(611)
    data = [rng.randint(-100, 100) for _ in range(10000)]
    before = data.copy()
    dc_result, dc_time = timed(lambda: max_subarray_dc(data), 5)
    kd_result, kd_time = timed(lambda: kadane_reduce(data), 5)
    assert dc_result == kd_result and data == before
    print("10 000 чисел; seed=611; медиана 5 запусков:")
    print(f"DC: {dc_time * 1000:.3f} мс; reduce: {kd_time * 1000:.3f} мс")
    print("Результаты и индексы совпадают:", dc_result == kd_result)
    print("Исходные данные сохранены:", data == before)
    return {
        "sum": result.total,
        "start": result.start,
        "stop": result.stop,
        "dc_ms": dc_time * 1000,
        "kadane_ms": kd_time * 1000,
    }


def demo_l1d() -> dict[str, Any]:
    """Measure median sort times on shared seeded data of three sizes."""
    print("Python", platform.python_version(), ";", platform.platform())
    print("Целые числа [-1 000 000; 1 000 000]; seed=606+n")
    print("На алгоритм 5 реальных запусков; медиана, миллисекунды.")
    print("Генерация данных и проверка результата вне измеряемого участка.")
    print("       n | merge_sort | quick_sort |   sorted | Проверка")
    rows = []
    for size in (1000, 10000, 100000):
        rng = random.Random(606 + size)
        xs = [rng.randint(-1000000, 1000000) for _ in range(size)]
        before = xs.copy()
        expected = sorted(xs)
        timings = []
        for algorithm in (merge_sort, quick_sort, sorted):
            result, seconds = timed(lambda: algorithm(xs), 5)
            assert result == expected and xs == before
            timings.append(seconds * 1000)
        rows.append(
            {
                "n": size,
                "merge_ms": timings[0],
                "quick_ms": timings[1],
                "sorted_ms": timings[2],
            }
        )
        print(
            f"{size:8d} | {timings[0]:10.3f} | {timings[1]:10.3f} |"
            f" {timings[2]:8.3f} | PASS"
        )
    print("Все результаты равны sorted; входные списки не изменились.")
    return {"rows": rows, "repeats": 5}


def demo_l2a() -> dict[str, Any]:
    """Flatten all JSON values into the sequence of atoms."""
    doc = load_document()
    before = copy.deepcopy(doc)
    flat = list(deep_flatten(doc))
    print("Загружен data/nested.json через json.loads.")
    print("deep_flatten, атомарные элементы по порядку:")
    for index, value in enumerate(flat):
        print(f"{index:2d}: {value!r}")
    print("Число атомов:", len(flat))
    print("Пустые контейнеры не дают атомов:", list(deep_flatten([[], {}])))
    print("Исходный JSON сохранён:", doc == before)
    return {"count": len(flat), "flat": flat}


def demo_l2b() -> dict[str, Any]:
    """Show the explicit nesting-depth convention and an eight-level JSON."""
    doc = load_document()
    print("Источник: data/nested.json")
    print("Атом имеет глубину 0; контейнер добавляет 1.")
    print("Пустой список:", deep_depth([]))
    print("Атом 17:", deep_depth(17))
    print("[1, [2, [3]]]:", deep_depth([1, [2, [3]]]))
    print("Глубина JSON-документа:", deep_depth(doc))
    print("Самая глубокая ветвь:")
    print("root / department / staff / [0] / projects / [0] / metrics / hours")
    print("Условие не менее четырёх уровней:", deep_depth(doc) >= 4)
    assert deep_depth(doc) >= 4
    return {"depth": deep_depth(doc)}


def demo_l2c() -> dict[str, Any]:
    """Map all numeric atoms and retain lists, tuples and dictionary keys."""
    doc = load_document()
    before = copy.deepcopy(doc)

    def double_number(value: Any) -> Any:
        return value * 2 if type(value) in (int, float) else value

    mapped = deep_map(double_number, doc)
    path = ("department", "staff", 0, "projects", 0, "metrics")
    print("Источник: data/nested.json; каждый числовой атом умножен на 2.")
    print("metrics до:", get_path(doc, path))
    print("metrics после:", get_path(mapped, path))
    print("semester:", doc["semester"], "->", mapped["semester"])
    print(
        "Строки и bool сохранены:",
        mapped["department"]["staff"][0]["name"],
        mapped["department"]["staff"][0]["active"],
    )
    mixed = [1, (2, {"x": 3})]
    print("Отдельная проверка кортежа:", deep_map(double_number, mixed))
    print("Тип вложенного кортежа:", type(deep_map(double_number, mixed)[1]).__name__)
    print("Исходный JSON сохранён:", doc == before)
    assert doc == before
    return {"semester_before": doc["semester"], "semester_after": mapped["semester"]}


def demo_l2d() -> dict[str, Any]:
    """Sum numeric JSON atoms while excluding text, null and booleans."""
    doc = load_document()
    print("Источник: data/nested.json")
    print("Числа: age 31 и 28, часы 4 + 6 + 8, рейтинг 4.5, семестр 6.")
    print("deep_sum(doc) =", deep_sum(doc))
    print("Ручная проверка =", 31 + 28 + 4 + 6 + 8 + 4.5 + 6)
    print("True/False исключены из суммы, несмотря на bool subclass int.")
    print("[1, '2', True, None, (3.5,)]:", deep_sum([1, "2", True, None, (3.5,)]))
    print("Пустая структура:", deep_sum({}))
    assert deep_sum(doc) == 87.5
    return {"sum": deep_sum(doc)}


def demo_l2e() -> dict[str, Any]:
    """List every leaf path from the JSON file and verify it by reading."""
    doc = load_document()
    leaves = list(paths(doc))
    print("Источник: data/nested.json; paths(doc):")
    for path, value in leaves:
        print(f"{path!r} = {value!r}")
    print("Число листьев, включая пустой список projects:", len(leaves))
    print(
        "Каждый путь проверен get_path:",
        all(get_path(doc, path) == value for path, value in leaves),
    )
    return {"leaves": len(leaves)}


def demo_l2f() -> dict[str, Any]:
    """Read a nested value and create a new version by immutable assignment."""
    doc = load_document()
    before = copy.deepcopy(doc)
    path = ("department", "staff", 0, "projects", 0, "metrics", "hours", 1)
    updated = set_path(doc, path, 12)
    print("Источник: data/nested.json")
    print("Путь:", path)
    print("get_path исходной версии:", get_path(doc, path))
    print("get_path новой версии:", get_path(updated, path))
    print("Исходная версия после set_path:", get_path(doc, path))
    print("Корни различаются:", updated is not doc)
    print(
        "Изменённая ветвь скопирована:", updated["department"] is not doc["department"]
    )
    print(
        "Неизменённый сотрудник разделяется:",
        updated["department"]["staff"][1] is doc["department"]["staff"][1],
    )
    print("Весь исходный JSON сохранён:", doc == before)
    print("Пустой путь заменяет корень:", set_path(doc, (), {"new": 1}))
    try:
        get_path(doc, ("missing",))
    except KeyError:
        print("Отсутствующий ключ: ожидаемый KeyError")
    assert doc == before and get_path(updated, path) == 12
    return {"before": get_path(doc, path), "after": get_path(updated, path)}


def demo_l4a() -> dict[str, Any]:
    """Generate all permutations and combinations without itertools."""
    ps = list(permutations([1, 2, 3]))
    cs = list(combinations([1, 2, 3, 4, 5], 3))
    print("Все перестановки [1, 2, 3]:")
    for item in ps:
        print(item)
    print("Число перестановок:", len(ps), "= 3! =", math.factorial(3))
    print("Все сочетания из [1, 2, 3, 4, 5] по 3:")
    for item in cs:
        print(item)
    print("Число сочетаний:", len(cs), "= C(5, 3) =", math.comb(5, 3))
    print("Пустая перестановка:", list(permutations([])))
    print("Сочетание по 0:", list(combinations([1, 2], 0)))
    print("Сочетание по 4 из 2:", list(combinations([1, 2], 4)))
    assert len(ps) == 6 and len(cs) == 10
    return {"permutations": len(ps), "combinations": len(cs)}


def demo_l4b() -> dict[str, Any]:
    """Demonstrate pruning on positive data and correctness on signed data."""
    positive = list(subset_sum([2, 3, 5, 7], 10))
    signed = list(subset_sum([8, -3, 2, 5], 5))
    zeros = list(subset_sum([0, 0], 0))
    print("[2, 3, 5, 7], сумма 10:", positive)
    print("Для неотрицательных чисел сумма > цели отсекается.")
    print("[8, -3, 2, 5], сумма 5:", signed)
    print("При наличии отрицательных чисел это отсечение отключено.")
    print("Пример 8 > 5, но 8 + (-3) = 5: найден", (8, -3) in signed)
    print("[0, 0], сумма 0:", zeros)
    print("Равные значения на разных позициях считаются разными выборами.")
    print("Пустой список, сумма 0:", list(subset_sum([], 0)))
    print("Решений нет:", list(subset_sum([2, 4], 3)))
    assert (8, -3) in signed and len(zeros) == 4
    return {"positive": positive, "signed": signed, "zero_count": len(zeros)}


def demo_l4c() -> dict[str, Any]:
    """Measure both queens solvers on n=4..10, median of three runs."""
    print("Python", platform.python_version(), "; медиана 3 реальных запусков.")
    print("Отсечение: битовые маски столбцов и диагоналей на каждом уровне.")
    print("Без раннего отсечения: все n! перестановок столбцов;")
    print("диагонали проверяются только на полностью заполненной доске.")
    print("Строки и уникальные столбцы заданы в обоих алгоритмах.")
    print(" n | решений | отсечение, мс | без раннего, мс | отношение")
    expected = (2, 10, 4, 40, 92, 352, 724)
    rows = []
    for n, known in zip(range(4, 11), expected):
        pruned, pruned_time = timed(lambda: queens_count(n), 3)
        unpruned, unpruned_time = timed(lambda: queens_count(n, prune=False), 3)
        assert pruned == unpruned == known
        ratio = unpruned_time / pruned_time
        row = {
            "n": n,
            "solutions": pruned,
            "pruned_ms": pruned_time * 1000,
            "unpruned_ms": unpruned_time * 1000,
            "ratio": ratio,
        }
        rows.append(row)
        print(
            f"{n:2d} | {pruned:7d} | {pruned_time * 1000:13.3f} |"
            f" {unpruned_time * 1000:14.3f} | {ratio:8.1f}x"
        )
    print("Все числа решений совпали в двух независимых режимах: PASS")
    return {"rows": rows, "repeats": 3}


def demo_l4d() -> dict[str, Any]:
    """Find a real maze path and check every transition independently."""
    maze = (
        "#########",
        "#S..#...#",
        "#.#.#.#E#",
        "#.#...#.#",
        "#.#####.#",
        "#.......#",
        "#########",
    )
    path = maze_path(maze)
    print("Лабиринт:")
    print("\n".join(maze))
    print("Путь (строка, столбец), индексы с нуля:")
    print(path)
    assert path is not None
    path_set = set(path)
    print("Найденный маршрут отмечен символом *:")
    for row, line in enumerate(maze):
        print(
            "".join(
                "*" if (row, col) in path_set and char == "." else char
                for col, char in enumerate(line)
            )
        )
    valid = (
        path[0] == (1, 1)
        and path[-1] == (2, 7)
        and len(set(path)) == len(path)
        and all(
            abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:])
        )
    )
    print("Клеток в пути:", len(path), "; переходов:", len(path) - 1)
    print("Проверка начала, конца, соседства и отсутствия циклов:", valid)
    print("Закрытый выход:", maze_path(("#####", "#S#E#", "#####")))
    print("DFS находит допустимый путь; кратчайший путь не обещается.")
    assert valid
    return {"cells": len(path), "steps": len(path) - 1, "path": path}


def specifications() -> list[dict[str, Any]]:
    """Provide separate report text for every requested bullet point."""
    return [
        {
            "id": "L1a",
            "title": "Задание 1.1. Сортировка слиянием и быстрая сортировка",
            "task": (
                "Реализовать сортировку слиянием и функциональную быструю сортировку с "
                "медианой из трёх."
            ),
            "explanation": [
                (
                    "В merge_sort я делю последовательность пополам, отдельно сортирую "
                    "обе части и объединяю их функцией merge. Базовый случай состоит "
                    "из нуля или одного элемента. При равенстве значений merge сначала "
                    "берёт элемент из левой части, поэтому порядок равных элементов "
                    "сохраняется. Входной список не изменяется, результат создаётся "
                    "отдельно."
                ),
                (
                    "В quick_sort опорное значение выбирается как медиана первого, "
                    "среднего и последнего элементов. Затем создаются три списка со "
                    "значениями меньше, равными и больше опорного. Рекурсия "
                    "применяется только к двум крайним спискам. Благодаря отдельной "
                    "группе равных элементов список из одинаковых значений завершается "
                    "за один уровень разбиения."
                ),
                (
                    "Для слияния T(n)=2T(n/2)+Θ(n), поэтому время составляет Θ(n log "
                    "n), стек имеет глубину O(log n), дополнительная память O(n). Для "
                    "быстрой сортировки T(n)=T(l)+T(r)+Θ(n), где l+r≤n−1. При "
                    "достаточно равномерном разбиении время O(n log n), в худшем "
                    "случае O(n²). Медиана трёх снижает риск неудачного выбора, но не "
                    "гарантирует баланс. Создание новых списков сохраняет "
                    "функциональный интерфейс; память в среднем O(n), в худшей цепочке "
                    "разбиений может достичь O(n²). Локальные append внутри merge не "
                    "затрагивают данные вызывающего кода и нужны для линейного слияния."
                ),
            ],
            "result": [],
            "code_files": ["lab06/dc.py"],
        },
        {
            "id": "L1b",
            "title": "Задание 1.2. Рекурсивный подсчёт инверсий",
            "task": (
                "Подсчитать инверсии за O(n log n) и сравнить с наивным алгоритмом на "
                "100 случайных списках."
            ),
            "explanation": [
                (
                    "Инверсия представляет собой пару индексов i<j, для которой левый "
                    "элемент больше правого. Я считаю инверсии внутри левой части, "
                    "внутри правой и между частями. Между частями подсчёт выполняется "
                    "во время слияния. Если очередной правый элемент меньше очередного "
                    "левого, он меньше всех оставшихся элементов отсортированной левой "
                    "части; к ответу добавляется len(left)−i."
                ),
                (
                    "Равные значения инверсию не образуют, поэтому при равенстве "
                    "сначала выбирается левый элемент. Рекуррентное соотношение "
                    "T(n)=2T(n/2)+Θ(n) даёт Θ(n log n). Наивная проверка рассматривает "
                    "каждую пару индексов и работает за Θ(n²). Она не использует код "
                    "слияния и служит независимым способом проверки."
                ),
                (
                    "Я создала генератор Random с фиксированным seed=610. Для каждого "
                    "из 100 списков длиной от 0 до 80 сравнила оба результата и "
                    "проверила сохранение исходного списка. Диапазон от −30 до 30 "
                    "специально допускает отрицательные и повторяющиеся значения."
                ),
            ],
            "result": [],
            "code_files": ["lab06/dc.py", "tests/test_algorithms.py"],
        },
        {
            "id": "L1c",
            "title": "Задание 1.3. Подмассив с максимальной суммой",
            "task": (
                "Реализовать метод «разделяй и властвуй» и сравнить с алгоритмом "
                "Кадане через reduce."
            ),
            "explanation": [
                (
                    "Искомый непустой подмассив может целиком находиться слева от "
                    "середины, целиком справа либо пересекать середину. Первые два "
                    "случая решаются рекурсивно. Для третьего я нахожу лучший суффикс "
                    "левой части и лучший префикс правой части. Из трёх кандидатов "
                    "выбирается наибольшая сумма."
                ),
                (
                    "Функция возвращает именованный кортеж Subarray с суммой, "
                    "начальным индексом и исключённой правой границей. При равных "
                    "суммах предпочтение получает меньший начальный индекс, затем "
                    "меньшая правая граница. Этот единый порядок нужен, чтобы "
                    "сравнивать алгоритмы не только по сумме, но и по выбранному "
                    "интервалу. Пустой вход вызывает ValueError. Если все числа "
                    "отрицательны, результатом остаётся один наименее отрицательный "
                    "элемент, а не пустой подмассив."
                ),
                (
                    "В Кадане состояние reduce состоит из лучшего интервала, "
                    "заканчивающегося на текущем элементе, и лучшего интервала за всё "
                    "пройденное время. На каждом шаге выбирается продолжение прошлого "
                    "интервала или новый интервал из одного элемента. Для рекурсивного "
                    "метода T(n)=2T(n/2)+Θ(n)=Θ(n log n), стек O(log n). Кадане "
                    "выполняет Θ(n) шагов и использует O(1) дополнительной памяти, "
                    "поскольку индексированные элементы подаются генератором. "
                    "Измерения выполнены на одном списке длиной 10 000, медиана пяти "
                    "запусков."
                ),
            ],
            "result": [],
            "code_files": ["lab06/dc.py"],
        },
        {
            "id": "L1d",
            "title": "Задание 1.4. Измерение времени сортировок",
            "task": (
                "Измерить сортировки на 10³, 10⁴ и 10⁵ элементах и сравнить со sorted."
            ),
            "explanation": [
                (
                    "Для каждой длины я создала один список случайных целых чисел и "
                    "передала тот же набор всем трём алгоритмам. Seed равен 606+n. "
                    "Время измеряется perf_counter, а итоговое значение представляет "
                    "собой медиану пяти запусков. Генерация данных и проверка "
                    "правильности не включаются в измеряемый участок."
                ),
                (
                    "После серии измеряемых запусков последний результат проверяется "
                    "по sorted, а входной список сравнивается с заранее сохранённой "
                    "копией. Все три алгоритма возвращают новый список, поэтому "
                    "копирование входа между запусками не требуется. Для sorted "
                    "измеряется сам вызов, включая создание результата, как и у двух "
                    "собственных функций."
                ),
                (
                    "Встроенная sorted использует оптимизированную реализацию Timsort. "
                    "Её преимущество во времени объясняется реализацией и меньшими "
                    "накладными расходами, а не иной оценкой худшего случая. На "
                    "случайных данных обе учебные сортировки ожидаемо ведут себя "
                    "близко к n log n. Конкретные миллисекунды относятся к этой машине "
                    "и этому запуску; они не являются универсальными константами."
                ),
            ],
            "result": [],
            "code_files": ["lab06/dc.py", "demo_algorithms.py"],
        },
        {
            "id": "L2a",
            "title": "Задание 2.1. Генератор deep_flatten",
            "task": "Рекурсивно выдать все атомарные элементы вложенной структуры.",
            "explanation": [
                (
                    "deep_flatten различает три вида узлов. У словаря она обходит "
                    "значения, у списка и кортежа последовательно обходит элементы, а "
                    "любой другой объект считает атомом и выдаёт через yield. "
                    "Рекурсивная передача результатов выполнена через yield from. "
                    "Строка остаётся одним атомом и не распадается на символы."
                ),
                (
                    "Для проверки я загрузила data/nested.json стандартным модулем "
                    "json. Порядок атомов соответствует порядку обхода исходных "
                    "списков и ключей словаря. Пустые контейнеры атомов не содержат, а "
                    "логические значения и None сохраняются как отдельные атомы. "
                    "Генератор не собирает полный результат заранее; list используется "
                    "только в демонстрации. При N узлах время O(N), стек O(h), где h "
                    "есть глубина вложенности."
                ),
            ],
            "result": [],
            "code_files": ["lab06/nested.py", "data/nested.json"],
        },
        {
            "id": "L2b",
            "title": "Задание 2.2. Максимальная глубина deep_depth",
            "task": "Определить максимальную глубину вложенности структуры.",
            "explanation": [
                (
                    "Я зафиксировала соглашение о глубине: атом имеет глубину 0, любой "
                    "контейнер добавляет один уровень. Поэтому пустой контейнер имеет "
                    "глубину 1. Для непустого контейнера функция возвращает 1 плюс "
                    "максимальную глубину его потомков. Это же правило применяется к "
                    "спискам, кортежам и значениям словарей."
                ),
                (
                    "В JSON самая глубокая ветвь проходит через корневой словарь, "
                    "department, staff, словарь первого сотрудника, projects, словарь "
                    "проекта, metrics и hours. Каждый из этих восьми контейнеров "
                    "добавляет уровень. Таким образом, исходные данные удовлетворяют "
                    "требованию не менее четырёх уровней. Функция посещает все N узлов "
                    "за O(N) и использует стек O(h)."
                ),
            ],
            "result": [],
            "code_files": ["lab06/nested.py", "data/nested.json"],
        },
        {
            "id": "L2c",
            "title": "Задание 2.3. Преобразование deep_map",
            "task": (
                "Применить функцию к атомам с сохранением списков, кортежей и значений "
                "словарей."
            ),
            "explanation": [
                (
                    "deep_map создаёт новый контейнер того же вида и рекурсивно "
                    "преобразует его потомков. Ключи словаря остаются прежними. "
                    "Пользовательская функция вызывается только для атомов. В "
                    "демонстрации числовые значения удваиваются, а строки, None и bool "
                    "возвращаются без изменений."
                ),
                (
                    "В загруженном JSON проверены вложенные часы и рейтинг проекта, "
                    "номер семестра, имя сотрудника и логический признак active. JSON "
                    "не содержит кортежей, поэтому отдельно обработана смешанная "
                    "структура [1, (2, {'x': 3})]. Полученный результат сохраняет "
                    "вложенный tuple. Исходные данные сравниваются с глубокой копией. "
                    "При O(1) времени пользовательской функции обход занимает O(N); "
                    "новый результат требует O(N) памяти и стек O(h). Чистота deep_map "
                    "предполагает, что переданная функция f тоже не изменяет атомы и "
                    "внешнее состояние."
                ),
            ],
            "result": [],
            "code_files": ["lab06/nested.py"],
        },
        {
            "id": "L2d",
            "title": "Задание 2.4. Сумма чисел deep_sum",
            "task": "Сложить все числовые атомы и проигнорировать нечисловые.",
            "explanation": [
                (
                    "deep_sum использует deep_flatten и фильтрует его атомы по классу "
                    "numbers.Number. Это позволяет складывать обычные целые и "
                    "вещественные числа. Числовые типы должны быть совместимы со "
                    "сложением Python. Логические значения исключаются явно: хотя bool "
                    "является подклассом int, в этом JSON true и false обозначают "
                    "признаки, а не числовые измерения."
                ),
                (
                    "Строки с цифрами не преобразуются в числа, None и пустые "
                    "контейнеры дают нулевой вклад. Я проверила сумму независимо, "
                    "выписав числовые значения документа. Для проверки границ также "
                    "использованы пустой словарь и смешанная структура со строкой '2' "
                    "и True. Обход занимает O(N), а вспомогательная память ограничена "
                    "стеком глубины O(h)."
                ),
            ],
            "result": [],
            "code_files": ["lab06/nested.py"],
        },
        {
            "id": "L2e",
            "title": "Задание 2.5. Генератор путей paths",
            "task": (
                "Выдать пары из кортежа ключей/индексов и соответствующего листового "
                "значения JSON."
            ),
            "explanation": [
                (
                    "Во время обхода paths передаёт накопленный путь в виде "
                    "неизменяемого кортежа. При переходе по словарю в путь добавляется "
                    "ключ, а при переходе по списку добавляется индекс. Когда "
                    "достигнут лист, генератор выдаёт пару (путь, значение). Для "
                    "атомарного корня путь был бы пустым кортежем."
                ),
                (
                    "Я считаю пустые массивы и объекты терминальными листьями. "
                    "Благодаря этому из вывода не исчезает поле projects второго "
                    "сотрудника, даже если список проектов пуст. Все полученные пути "
                    "независимо проверены вызовом get_path. Формирование кортежа пути "
                    "копирует его префикс, поэтому время и объём выдаваемых путей "
                    "зависят от суммарной длины путей, вплоть до O(Nh). Генератор не "
                    "хранит весь список результатов; активные кортежи пути в стеке в "
                    "худшем случае требуют O(h²) памяти."
                ),
            ],
            "result": [],
            "code_files": ["lab06/nested.py", "data/nested.json"],
        },
        {
            "id": "L2f",
            "title": "Задание 2.6. Чтение и неизменяющая запись по пути",
            "task": (
                "Реализовать get_path и set_path, сохраняя прежнюю версию документа."
            ),
            "explanation": [
                (
                    "get_path последовательно выбирает потомка по компонентам пути. "
                    "set_path идёт по тому же маршруту, а при возврате из рекурсии "
                    "создаёт новые контейнеры только на изменяемой ветви. Другие ветви "
                    "сохраняют прежние ссылки. Такое разделение структуры позволяет "
                    "одновременно иметь старую и новую версии без полного копирования "
                    "документа."
                ),
                (
                    "В демонстрации второй элемент hours меняется с 6 на 12. Чтение "
                    "старой версии после операции по-прежнему даёт 6. Проверка "
                    "идентичности показывает, что корень и department скопированы, а "
                    "словарь второго сотрудника разделяется между версиями. "
                    "Неизменность гарантирует сама операция; поскольку Python list и "
                    "dict изменяемы, последующие внешние изменения разделяемых ветвей "
                    "пользователь должен исключить."
                ),
                (
                    "Пустой путь означает обращение к корню или замену корня. Функции "
                    "работают с существующими путями: неизвестный ключ вызывает "
                    "KeyError, выход за границы списка вызывает IndexError, попытка "
                    "пройти внутрь атома вызывает TypeError. Индексы массивов должны "
                    "быть неотрицательными целыми числами. В этой простой рекурсивной "
                    "реализации срезы path[1:] дают до O(h²) затрат на путь. Для "
                    "set_path дополнительно требуется скопировать элементы контейнеров "
                    "на пути, то есть O(Σbᵢ), где bᵢ есть размер соответствующего "
                    "контейнера."
                ),
            ],
            "result": [],
            "code_files": ["lab06/nested.py"],
        },
        {
            "id": "L4a",
            "title": "Задание 4.1. Перестановки и сочетания",
            "task": (
                "Сгенерировать все перестановки и сочетания из n по k без itertools."
            ),
            "explanation": [
                (
                    "Для перестановок я выбираю по очереди каждый элемент первым, "
                    "удаляю именно его позицию из доступного кортежа и рекурсивно "
                    "переставляю остальные. Пустой остаток даёт одну пустую "
                    "перестановку. Это базовый случай, к которому постепенно "
                    "приписываются выбранные элементы."
                ),
                (
                    "Для сочетаний элементы берутся только из хвоста после выбранной "
                    "позиции. Поэтому разные порядки одних и тех же позиций не "
                    "повторяются. При k=0 выдаётся пустое сочетание, при k>n решений "
                    "нет, отрицательное k считается ошибкой. Алгоритмы рассматривают "
                    "позиции, поэтому равные значения на разных позициях могут дать "
                    "одинаковые по значению результаты."
                ),
                (
                    "Вывести n! перестановок длины n требует как минимум Ω(n·n!) "
                    "времени. Вывод сочетаний требует Ω(k·C(n,k)). Создание срезов и "
                    "кортежей добавляет накладные расходы; генераторы выдают решения "
                    "последовательно, а не хранят весь набор. Демонстрация специально "
                    "преобразует их в списки, чтобы показать все 6 перестановок и 10 "
                    "сочетаний и сверить количества с формулами."
                ),
            ],
            "result": [],
            "code_files": ["lab06/backtracking.py"],
        },
        {
            "id": "L4b",
            "title": "Задание 4.2. Подмножества с заданной суммой",
            "task": (
                "Найти все подмножества с целевой суммой и применять корректное "
                "отсечение."
            ),
            "explanation": [
                (
                    "На каждом шаге рекурсия принимает одно из двух решений: "
                    "пропустить очередной элемент или включить его в выбранный кортеж. "
                    "В листе дерева поиска сумма сравнивается с целью. Решения "
                    "проверяются именно в листе, чтобы не потерять продолжения с "
                    "нулями и отрицательными числами."
                ),
                (
                    "Если все числа неотрицательны, сумма не может уменьшиться при "
                    "добавлении элементов. Поэтому ветвь с текущей суммой больше "
                    "целевой можно сразу отбросить. При наличии отрицательных чисел "
                    "это правило отключается. Например, после выбора 8 сумма "
                    "превосходит цель 5, но последующее добавление −3 даёт правильный "
                    "ответ."
                ),
                (
                    "Пустое подмножество входит в ответы при цели 0. Повторяющиеся "
                    "значения считаются различными элементами по позиции; поэтому для "
                    "[0, 0] возникают четыре выбора, в том числе два одинаковых по "
                    "значению одноэлементных кортежа. В худшем случае перебираются 2ⁿ "
                    "подмножеств; выдача и копирование кортежей дают до O(n·2ⁿ) "
                    "времени. Отсечение ускоряет подходящие наборы, но не меняет "
                    "экспоненциальную худшую оценку."
                ),
            ],
            "result": [],
            "code_files": ["lab06/backtracking.py"],
        },
        {
            "id": "L4c",
            "title": "Задание 4.3. N ферзей и эффект отсечения",
            "task": (
                "Найти числа решений для n=4…10 и измерить время с отсечением и без "
                "раннего отсечения."
            ),
            "explanation": [
                (
                    "Оба режима размещают по одному ферзю в строке и не допускают "
                    "повторения столбцов. В режиме отсечения целочисленные битовые "
                    "маски обозначают занятые столбцы и диагонали. На следующий "
                    "уровень передаются новые значения масок. Если свободных клеток "
                    "нет, ветвь заканчивается сразу."
                ),
                (
                    "Для сравнения без раннего отсечения я рекурсивно перебираю все n! "
                    "перестановок столбцов. Диагонали проверяются только после "
                    "заполнения всей доски. Это не перебор всех nⁿ размещений, потому "
                    "что уникальность столбцов уже заложена в представление. Внутри "
                    "финальной проверки можно закончить проверку при первом конфликте, "
                    "но это не отсекает частично построенные доски."
                ),
                (
                    "Каждый режим действительно выполнен три раза для каждого n от 4 "
                    "до 10. В таблице приведены медианы perf_counter; перед фиксацией "
                    "строки оба результата сверяются между собой и с контрольным "
                    "числом решений. Полный перебор листьев имеет факториальный рост, "
                    "а проверка одной доски в худшем случае требует O(n²). Раннее "
                    "отсечение уменьшает число посещённых состояний. Измеренное "
                    "отношение времён также включает различия представления состояния "
                    "и проверки ограничений, поэтому это сравнение реализаций, а не "
                    "изолированная стоимость одной проверки."
                ),
            ],
            "result": [],
            "code_files": ["lab06/backtracking.py", "demo_algorithms.py"],
        },
        {
            "id": "L4d",
            "title": "Задание 4.4. Поиск пути в лабиринте",
            "task": "Вернуть путь от S к E как кортеж координат либо None.",
            "explanation": [
                (
                    "Лабиринт представлен неизменяемым кортежем строк. Функция "
                    "проверяет прямоугольную форму, допустимые символы и наличие ровно "
                    "одного старта и одного выхода. Координаты записываются как пара "
                    "(строка, столбец) с индексацией от нуля."
                ),
                (
                    "Поиск в глубину пробует соседей в порядке вправо, вниз, влево, "
                    "вверх. Стена, выход за границу и уже посещённая в текущем пути "
                    "клетка прекращают данную попытку. Для нового шага создаются новый "
                    "кортеж пути и новое frozenset посещённых клеток. Если выбранное "
                    "продолжение не приводит к выходу, функция возвращается к "
                    "предыдущему шагу и пробует следующего соседа."
                ),
                (
                    "Я проверила начало и конец найденного маршрута, соседство каждой "
                    "пары последовательных клеток и отсутствие повторов. Отдельный "
                    "лабиринт с закрытым выходом возвращает None. Метод находит первый "
                    "допустимый путь и не обещает минимальную длину. Посещения "
                    "хранятся отдельно для каждой ветви, поэтому одна клетка может "
                    "изучаться в разных путях. В худшем случае такой перебор простых "
                    "путей экспоненциален; копирование пути и множества сохраняет "
                    "внешний интерфейс чистым, но требует до O(V²) памяти активной "
                    "рекурсии при V проходимых клетках."
                ),
            ],
            "result": [],
            "code_files": ["lab06/backtracking.py"],
        },
    ]


def main() -> None:
    """Run one evidence section or all, optionally saving output and prose."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", default="all")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    demos = (
        demo_l1a,
        demo_l1b,
        demo_l1c,
        demo_l1d,
        demo_l2a,
        demo_l2b,
        demo_l2c,
        demo_l2d,
        demo_l2e,
        demo_l2f,
        demo_l4a,
        demo_l4b,
        demo_l4c,
        demo_l4d,
    )
    descriptions = specifications()
    valid_ids = [item["id"] for item in descriptions]
    if args.section != "all" and args.section not in valid_ids:
        parser.error("Unknown section; choose " + ", ".join(valid_ids))
    evidence = []
    for spec, demo in zip(descriptions, demos):
        if args.section != "all" and args.section != spec["id"]:
            continue
        print(f"[{spec['id']}] {spec['title']}", flush=True)
        captured = io.StringIO()
        with redirect_stdout(captured):
            measured = demo()
        output = captured.getvalue()
        print(output, flush=True)
        evidence.append(
            {
                "id": spec["id"],
                "title": spec["title"],
                "command": "python demo_algorithms.py --section " + spec["id"],
                "output": output,
                "measurements": measured,
            }
        )
        spec["result"] = result_paragraphs(spec["id"], measured)
    if args.record:
        root = Path(__file__).resolve().parent
        (root / "evidence_algorithms.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        selected = [
            item
            for item in descriptions
            if item["id"] in {entry["id"] for entry in evidence}
        ]
        (root / "content_algorithms.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Сохранены evidence_algorithms.json и content_algorithms.json")


def result_paragraphs(section: str, data: dict[str, Any]) -> list[str]:
    """Describe only values obtained by the executed demonstration."""
    if section == "L1a":
        return [
            f"Обе сортировки вернули {data['sorted']}. Проверка по sorted пройдена. "
            "Пустые списки и повторяющиеся значения обработаны; исходный список "
            "сохранён."
        ]
    if section == "L1b":
        return [
            "Для [2, 4, 1, 3, 5] найдены три инверсии: пары значений (2, 1), "
            "(4, 1) и (4, 3). Совпадение с независимым наивным алгоритмом получено "
            f"на {data['random_comparisons']} из 100 случайных списков. Для десяти "
            "убывающих значений получено 45 инверсий, что равно 10·9/2."
        ]
    if section == "L1c":
        return [
            f"Оба метода выбрали [4, −1, 2, 1] с суммой {data['sum']}, "
            f"индексы [{data['start']}:{data['stop']}]. На 10 000 числах медиана "
            f"пяти запусков составила {data['dc_ms']:.3f} мс для рекурсивного "
            f"метода и {data['kadane_ms']:.3f} мс для Кадане. Результаты совпали "
            "полностью, включая границы интервала; входные данные сохранены."
        ]
    if section == "L1d":
        return [
            f"n={row['n']}: merge_sort {row['merge_ms']:.3f} мс; "
            f"quick_sort {row['quick_ms']:.3f} мс; sorted {row['sorted_ms']:.3f} мс. "
            "Проверка результата серии запусков и неизменности входа пройдена."
            for row in data["rows"]
        ]
    if section == "L2a":
        return [
            f"Генератор выдал {data['count']} атомов. В их числе сохранились "
            "строки, логические признаки и None. Пустой projects не создаёт атомов. "
            "Исходный документ совпал с сохранённой копией."
        ]
    if section == "L2b":
        return [
            f"Глубина загруженного JSON равна {data['depth']}. Для числа 17 "
            "получено 0, для пустого списка 1, для [1, [2, [3]]] получено 3."
        ]
    if section == "L2c":
        return [
            "Часы [4, 6, 8] преобразованы в [8, 12, 16], рейтинг 4.5 стал 9.0. "
            f"Номер семестра изменился с {data['semester_before']} на "
            f"{data['semester_after']}. Имя и логический признак остались прежними. "
            "Смешанный пример сохранил tuple, входной JSON не изменился."
        ]
    if section == "L2d":
        return [
            f"Получена сумма {data['sum']}. Независимый подсчёт "
            "31+28+4+6+8+4.5+6 дал тот же результат. В смешанном примере "
            "[1, '2', True, None, (3.5,)] получено 4.5, для пустого словаря 0."
        ]
    if section == "L2e":
        return [
            f"Выдано {data['leaves']} листовых путей. Для каждого пути "
            "get_path возвратил именно выданное значение. Пустой список projects "
            "сохранён как терминальное значение, поэтому число листьев на один "
            "больше числа атомов deep_flatten."
        ]
    if section == "L2f":
        return [
            f"В новой версии значение hours[1] равно {data['after']}, в старой "
            f"по-прежнему {data['before']}. Проверки различия изменяемых контейнеров "
            "и разделения неизменённой ветви вернули True. Отсутствующий ключ "
            "вызвал ожидаемый KeyError, замена корня по пустому пути также проверена."
        ]
    if section == "L4a":
        return [
            f"Получены все {data['permutations']} перестановок трёх элементов "
            f"и все {data['combinations']} сочетаний из пяти по три. Количества "
            "равны 3!=6 и C(5,3)=10. Пустая перестановка и сочетание по нулю "
            "представлены одним пустым кортежем, сочетаний по четыре из двух нет."
        ]
    if section == "L4b":
        return [
            f"Для [2, 3, 5, 7] с целью 10 получены {data['positive']}. "
            f"Для [8, −3, 2, 5] с целью 5 получены {data['signed']}; "
            "решение с 8 и −3 не потеряно. Для двух нулей и цели 0 выданы все "
            f"{data['zero_count']} позиционных выбора."
        ]
    if section == "L4c":
        lines = [
            f"n={row['n']}, решений {row['solutions']}; с отсечением "
            f"{row['pruned_ms']:.3f} мс, без раннего отсечения "
            f"{row['unpruned_ms']:.3f} мс, отношение времён {row['ratio']:.1f}."
            for row in data["rows"]
        ]
        return lines + [
            "Числа решений для n=4…10 равны 2, 10, 4, 40, 92, 352, "
            "724. Каждый результат получен обоими режимами, все проверки совпали."
        ]
    return [
        f"Найден маршрут из {data['cells']} клеток, то есть {data['steps']} "
        "переходов. Начало, конец, соседство и отсутствие повторных клеток "
        "проверены программно. Для лабиринта с закрытым выходом получено None."
    ]


if __name__ == "__main__":
    main()
