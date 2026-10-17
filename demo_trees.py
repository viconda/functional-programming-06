"""Запускаемые демонстрации заданий 3, 5, 6 и АВЛ-дерева."""

from __future__ import annotations

import argparse
import json
from contextlib import redirect_stdout
from functools import reduce
from io import StringIO
from math import log2
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any

from lab06 import avl, bst, fs_tree, genealogy


def run_section(section: str) -> None:
    """Реально выполнить выбранный подпункт и напечатать его результат."""
    tree = reduce(bst.insert, (50, 30, 70, 20, 40, 60, 80, 35), None)
    filesystem = fs_tree.sample_tree()
    family = genealogy.sample_family()
    if section == "L3example":
        second = bst.delete(tree, 30)
        third = bst.insert(second, 65)
        for name, version in (("v1", tree), ("v2", second), ("v3", third)):
            print(
                name,
                list(bst.inorder(version)),
                "высота",
                bst.height(version),
                "BST:",
                bst.is_bst(version),
            )
        print("Минимум v1:", bst.minimum(tree))
        print("Правое поддерево v1 и v2 разделяется:", tree.right is second.right)
        print("Левое поддерево v2 и v3 разделяется:", second.left is third.left)
    elif section == "L3a":
        print("preorder:", list(bst.preorder(tree)))
        print("postorder:", list(bst.postorder(tree)))
        print("level_order:", list(bst.level_order(tree)))
        print("Исходный inorder:", list(bst.inorder(tree)))
    elif section == "L3b":
        keys = list(range(1, 16))
        result = bst.from_sorted(keys)
        print("Вход:", keys)
        print("Обход по уровням:", list(bst.level_order(result)))
        print("Высота:", bst.height(result), "; корень:", result.key)
        print("Исходный список сохранился:", keys == list(range(1, 16)))
    elif section == "L3c":
        balanced = bst.from_sorted(range(1, 16))
        chain = reduce(bst.insert, range(1, 8), None)
        print("from_sorted(1..15), баланс:", bst.is_balanced(balanced))
        print("Последовательная вставка 1..7, баланс:", bst.is_balanced(chain))
        print("Пустое дерево сбалансировано:", bst.is_balanced(None))
    elif section == "L3d":
        for lo, hi in ((35, 65), (40, 40), (81, 100), (80, 20)):
            print(f"range_query({lo}, {hi}):", list(bst.range_query(tree, lo, hi)))
    elif section == "L3e":
        for k in (1, 4, 8):
            print(f"kth_smallest(k={k}):", bst.kth_smallest(tree, k))
        for k in (0, 9):
            try:
                bst.kth_smallest(tree, k)
            except IndexError as error:
                print(f"k={k}: {type(error).__name__}: {error}")
    elif section == "L3f":
        shuffled = list(range(1000))
        Random(306).shuffle(shuffled)
        for name, keys in (
            ("Возрастающий порядок", range(1000)),
            ("Случайный порядок, seed=306", shuffled),
        ):
            result = reduce(bst.insert_iterative, keys, None)
            print(
                f"{name}: n=1000, высота={bst.height(result)}, BST={bst.is_bst(result)}"
            )
        result = bst.from_sorted(range(1000))
        print(
            f"from_sorted: n=1000, высота={bst.height(result)}, "
            f"баланс={bst.is_balanced(result)}"
        )
        print("Высота измерена в уровнях; пустое дерево имеет высоту 0.")
    elif section == "L5a":
        print("Общий размер /project:", fs_tree.total_size(filesystem), "байт")
        print("Пустой каталог:", fs_tree.total_size(("dir", "empty", ())), "байт")
        print("Размеры файлов:", [size for _, size in fs_tree.files(filesystem)])
    elif section == "L5b":
        for suffix in (".py", ".md", "", ".cpp"):
            found = list(fs_tree.find_extension(filesystem, suffix))
            print(f"Расширение {suffix!r}:", found)
    elif section == "L5c":
        print("Самый большой файл:", fs_tree.largest_file(filesystem))
        print("Пустой каталог:", fs_tree.largest_file(("dir", "empty", ())))
    elif section == "L5d":
        print(fs_tree.tree_text(filesystem))
    elif section == "L5e":
        report = fs_tree.extension_report(filesystem)
        for suffix, (count, size) in sorted(report.items()):
            print(f"{suffix or '[без расширения]'}: файлов={count}, байт={size}")
        print("Всего файлов:", sum(count for count, _ in report.values()))
        combined_size = sum(size for _, size in report.values())
        print("Сверка размеров:", combined_size == fs_tree.total_size(filesystem))
    elif section == "L6a":
        print("Предки Елены до 3-го поколения:")
        for level, names in genealogy.ancestors(family, "Елена", 3):
            print(f"Поколение {level}: {', '.join(names)}")
        print("Нулевое число поколений:", genealogy.ancestors(family, "Елена", 0))
        print(
            "У Петра нет известных родителей:",
            genealogy.ancestors(family, "Пётр", 3),
        )
    elif section == "L6b":
        for first, second in (
            ("Анна", "Сергей"),
            ("Анна", "Дмитрий"),
            ("Анна", "Ирина"),
            ("Анна", "Пётр"),
            ("Анна", "Анна"),
        ):
            shared = genealogy.common_ancestors(family, first, second)
            print(f"{first} + {second}:", shared)
        print("Критерий близости: сначала max(d1, d2), затем d1+d2.")
        print(
            "Все равные лучшие предки возвращаются; человек имеет расстояние 0 до себя."
        )
    elif section == "L6c":
        for name in ("Алексей", "Ирина", "Елена"):
            count = genealogy.descendant_count(family, name)
            print(f"{name}: {count} потомков", genealogy.descendants(family, name))
        print("Елена достижима по двум ветвям, но учитывается один раз.")
        for label, action in (
            ("Неизвестный человек", lambda: genealogy.descendants(family, "Нет")),
            (
                "Цикл A-B-A",
                lambda: genealogy.validate(
                    (
                        genealogy.Person("A", ("B",)),
                        genealogy.Person("B", ("A",)),
                    )
                ),
            ),
        ):
            try:
                action()
            except (KeyError, ValueError) as error:
                print(f"{label}: {type(error).__name__}: {error}")
    elif section == "L7":
        for rotation, keys in (
            ("LL", (3, 2, 1)),
            ("RR", (1, 2, 3)),
            ("LR", (3, 1, 2)),
            ("RL", (1, 3, 2)),
        ):
            result = reduce(avl.insert, keys, None)
            print(
                f"{rotation}: корень={result.key}, "
                f"обход={list(avl.inorder(result))}, AVL={avl.is_avl(result)}"
            )
        for size in (1000, 10000):
            start = perf_counter()
            result = reduce(avl.insert, range(size), None)
            elapsed = perf_counter() - start
            print(
                f"AVL: n={size}, h={avl.height(result)}, "
                f"log2(n+1)={log2(size + 1):.3f}, секунд={elapsed:.6f}, "
                f"AVL={avl.is_avl(result)}"
            )
        old_ids = avl.node_ids(result)
        updated = avl.insert(result, 10000)
        new_ids = avl.node_ids(updated)
        print(
            f"Вставка 10000: новых сохранённых узлов={len(new_ids - old_ids)}, "
            f"общих узлов={len(new_ids & old_ids)}"
        )
        print("Старая версия содержит", len(old_ids), "ключей; новая", len(new_ids))
        start = perf_counter()
        plain = reduce(bst.insert_iterative, range(1000), None)
        elapsed = perf_counter() - start
        print(
            f"Обычный BST: n=1000, h={bst.height(plain)}, "
            f"секунд={elapsed:.6f}, BST={bst.is_bst(plain)}"
        )
        print("Для обычного BST сравнение ограничено n=1000; вставка итеративная.")
    else:
        raise ValueError(f"Неизвестный раздел: {section}")


def sections() -> tuple[tuple[str, str], ...]:
    """Полный список отдельных подпунктов для отчёта."""
    return (
        ("L3example", "Пример: вставка, удаление и сохранение версий"),
        ("L3a", "Задание 3.1. Три обхода дерева"),
        ("L3b", "Задание 3.2. Построение from_sorted"),
        ("L3c", "Задание 3.3. Проверка сбалансированности"),
        ("L3d", "Задание 3.4. Запрос диапазона"),
        ("L3e", "Задание 3.5. k-й наименьший ключ"),
        ("L3f", "Задание 3.6. Сравнение высот при n=1000"),
        ("L5a", "Задание 5.1. Общий размер каталога"),
        ("L5b", "Задание 5.2. Поиск по расширению"),
        ("L5c", "Задание 5.3. Самый большой файл"),
        ("L5d", "Задание 5.4. Дерево каталогов с отступами"),
        ("L5e", "Задание 5.5. Отчёт по расширениям"),
        ("L6a", "Вариант 3.1. Предки до n-го поколения"),
        ("L6b", "Вариант 3.2. Ближайшие общие предки"),
        ("L6c", "Вариант 3.3. Число потомков"),
        ("L7", "Дополнительное задание. Неизменяемое АВЛ-дерево"),
    )


def evidence() -> list[dict[str, Any]]:
    """Снять фактический текст каждого запуска без имитации результатов."""
    result = []
    for identifier, title in sections():
        output = StringIO()
        with redirect_stdout(output):
            run_section(identifier)
        result.append(
            {
                "id": identifier,
                "title": title,
                "command": f"python demo_trees.py --section {identifier}",
                "output": output.getvalue(),
            }
        )
    return result


def main() -> None:
    """Оболочка ввода-вывода; вычислительные модули сами не печатают."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=[key for key, _ in sections()])
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    if args.section:
        run_section(args.section)
    else:
        captured = evidence()
        for item in captured:
            print(item["title"])
            print(item["output"])
        if args.evidence:
            args.evidence.write_text(
                json.dumps(captured, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
