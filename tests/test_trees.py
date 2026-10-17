"""Проверки деревьев, файловой модели и родословной."""

import unittest
from dataclasses import FrozenInstanceError
from functools import reduce
from random import Random

from lab06 import avl, bst, fs_tree, genealogy


class BstTests(unittest.TestCase):
    """Проверки функционального BST, включая вырожденный случай."""

    def setUp(self) -> None:
        self.tree = reduce(bst.insert, (50, 30, 70, 20, 40, 60, 80, 35), None)

    def test_versions_and_sharing(self) -> None:
        second = bst.delete(self.tree, 30)
        third = bst.insert(second, 65)
        self.assertEqual(list(bst.inorder(self.tree)), [20, 30, 35, 40, 50, 60, 70, 80])
        self.assertEqual(list(bst.inorder(second)), [20, 35, 40, 50, 60, 70, 80])
        self.assertEqual(list(bst.inorder(third)), [20, 35, 40, 50, 60, 65, 70, 80])
        self.assertIs(self.tree.right, second.right)
        self.assertIs(second.left, third.left)
        self.assertEqual(bst.minimum(self.tree), 20)
        self.assertEqual([bst.height(t) for t in (self.tree, second, third)], [4, 3, 4])
        self.assertTrue(all(bst.is_bst(t) for t in (self.tree, second, third)))

    def test_duplicates_and_absent_delete_preserve_identity(self) -> None:
        self.assertIs(bst.insert(self.tree, 35), self.tree)
        self.assertIs(bst.insert_iterative(self.tree, 35), self.tree)
        self.assertIs(bst.delete(self.tree, -9), self.tree)

    def test_delete_leaf_single_child_two_children_and_root(self) -> None:
        for key in (20, 40, 30, 50):
            result = bst.delete(self.tree, key)
            self.assertEqual(
                list(bst.inorder(result)),
                sorted(set(bst.inorder(self.tree)) - {key}),
            )
            self.assertTrue(bst.is_bst(result))
        self.assertIsNone(bst.delete(bst.Node(1), 1))
        self.assertIsNone(bst.delete(None, 1))

    def test_traversals(self) -> None:
        self.assertEqual(
            list(bst.preorder(self.tree)),
            [50, 30, 20, 40, 35, 70, 60, 80],
        )
        self.assertEqual(
            list(bst.postorder(self.tree)),
            [20, 35, 40, 30, 60, 80, 70, 50],
        )
        self.assertEqual(
            list(bst.level_order(self.tree)),
            [50, 30, 70, 20, 40, 60, 80, 35],
        )
        for traversal in (bst.inorder, bst.preorder, bst.postorder, bst.level_order):
            self.assertEqual(list(traversal(None)), [])

    def test_from_sorted_and_balance(self) -> None:
        tree = bst.from_sorted(range(1000))
        self.assertEqual(bst.height(tree), 10)
        self.assertTrue(bst.is_balanced(tree))
        self.assertEqual(list(bst.inorder(tree)), list(range(1000)))
        self.assertIsNone(bst.from_sorted([]))
        self.assertTrue(bst.is_balanced(None))
        self.assertFalse(bst.is_balanced(reduce(bst.insert, range(5), None)))
        for keys in ([1, 1], [2, 1]):
            with self.assertRaises(ValueError):
                bst.from_sorted(keys)

    def test_range_and_kth(self) -> None:
        self.assertEqual(list(bst.range_query(self.tree, 35, 65)), [35, 40, 50, 60])
        self.assertEqual(list(bst.range_query(self.tree, 40, 40)), [40])
        self.assertEqual(list(bst.range_query(self.tree, 80, 20)), [])
        self.assertEqual(list(bst.range_query(None, 0, 9)), [])
        for k, value in enumerate(sorted(bst.inorder(self.tree)), 1):
            self.assertEqual(bst.kth_smallest(self.tree, k), value)
        for k in (0, -1, 9):
            with self.assertRaises(IndexError):
                bst.kth_smallest(self.tree, k)

    def test_iterative_depth_1000_and_invalid_tree(self) -> None:
        tree = reduce(bst.insert_iterative, range(1000), None)
        self.assertEqual(bst.height(tree), 1000)
        self.assertTrue(bst.is_bst(tree))
        self.assertEqual(bst.kth_smallest(tree, 1000), 999)
        self.assertFalse(bst.is_bst(bst.Node(5, bst.Node(4, right=bst.Node(6)))))

    def test_frozen_and_input_unchanged(self) -> None:
        keys = [1, 2, 3]
        bst.from_sorted(keys)
        self.assertEqual(keys, [1, 2, 3])
        with self.assertRaises(FrozenInstanceError):
            self.tree.key = 0

    def test_random_insert_delete_against_set(self) -> None:
        rng = Random(306)
        tree = None
        expected: set[int] = set()
        for _ in range(250):
            value = rng.randrange(100)
            if rng.random() < 0.6:
                tree = bst.insert(tree, value)
                expected.add(value)
            else:
                tree = bst.delete(tree, value)
                expected.discard(value)
            self.assertEqual(list(bst.inorder(tree)), sorted(expected))
            self.assertTrue(bst.is_bst(tree))


class FileSystemTests(unittest.TestCase):
    """Каждая из пяти операций и пустые структуры."""

    def test_size_and_paths(self) -> None:
        tree = fs_tree.sample_tree()
        self.assertEqual(fs_tree.total_size(tree), 19700)
        self.assertEqual(
            list(fs_tree.find_extension(tree, "py")),
            [
                "/project/src/main.py",
                "/project/src/utils/helpers.PY",
            ],
        )
        self.assertEqual(list(fs_tree.find_extension(tree, "")), ["/project/LICENSE"])
        self.assertEqual(list(fs_tree.find_extension(tree, ".cpp")), [])

    def test_largest(self) -> None:
        self.assertEqual(
            fs_tree.largest_file(fs_tree.sample_tree()),
            ("/project/docs/report.pdf", 8500),
        )
        tied = ("dir", "x", (("file", "a", 1), ("file", "b", 1)))
        self.assertEqual(fs_tree.largest_file(tied), ("/x/a", 1))

    def test_text_and_extensions(self) -> None:
        tree = fs_tree.sample_tree()
        self.assertIn("      helpers.PY (1600 байт)", fs_tree.tree_text(tree))
        report = fs_tree.extension_report(tree)
        self.assertEqual(report[".py"], (2, 4000))
        self.assertEqual(report[".md"], (2, 1900))
        self.assertEqual(report[""], (1, 500))
        self.assertEqual(sum(n for n, _ in report.values()), 9)
        self.assertEqual(
            sum(size for _, size in report.values()),
            fs_tree.total_size(tree),
        )

    def test_empty_and_immutability(self) -> None:
        empty = ("dir", "empty", ())
        self.assertEqual(fs_tree.total_size(empty), 0)
        self.assertIsNone(fs_tree.largest_file(empty))
        self.assertEqual(fs_tree.extension_report(empty), {})
        self.assertEqual(fs_tree.tree_text(empty), "empty/")
        original = fs_tree.sample_tree()
        before = repr(original)
        list(fs_tree.files(original))
        fs_tree.extension_report(original)
        self.assertEqual(repr(original), before)


class GenealogyTests(unittest.TestCase):
    """Проверки двух родителей, общих ветвей и некорректных данных."""

    def test_ancestors_and_generations(self) -> None:
        family = genealogy.sample_family()
        self.assertEqual(
            genealogy.ancestors(family, "Елена", 1),
            ((1, ("Анна", "Сергей")),),
        )
        self.assertEqual(
            genealogy.ancestors(family, "Елена", 3)[2],
            (3, ("Алексей", "Нина")),
        )
        self.assertEqual(genealogy.ancestors(family, "Елена", 0), ())
        self.assertEqual(genealogy.ancestors(family, "Пётр", 4), ())
        with self.assertRaises(ValueError):
            genealogy.ancestors(family, "Елена", -1)

    def test_common_ancestor_ties_self_and_absent(self) -> None:
        family = genealogy.sample_family()
        self.assertEqual(
            genealogy.common_ancestors(family, "Анна", "Сергей"),
            ("Алексей", "Нина"),
        )
        self.assertEqual(
            genealogy.common_ancestors(family, "Анна", "Дмитрий"),
            ("Ирина", "Олег"),
        )
        self.assertEqual(
            genealogy.common_ancestors(family, "Анна", "Ирина"),
            ("Ирина",),
        )
        self.assertEqual(genealogy.common_ancestors(family, "Анна", "Анна"), ("Анна",))
        self.assertEqual(genealogy.common_ancestors(family, "Анна", "Пётр"), ())

    def test_descendants_distinct(self) -> None:
        family = genealogy.sample_family()
        result = genealogy.descendants(family, "Алексей")
        self.assertEqual(len(result), 6)
        self.assertEqual(result.count("Елена"), 1)
        self.assertEqual(genealogy.descendant_count(family, "Елена"), 0)

    def test_unknown_and_invalid_graphs(self) -> None:
        family = genealogy.sample_family()
        with self.assertRaises(KeyError):
            genealogy.ancestors(family, "Нет", 0)
        with self.assertRaises(KeyError):
            genealogy.common_ancestors(family, "Нет", "Анна")
        with self.assertRaises(KeyError):
            genealogy.descendant_count(family, "Нет")
        invalid = (
            (genealogy.Person("A", ("B",)), genealogy.Person("B", ("A",))),
            (genealogy.Person("A", ("B",)),),
            (genealogy.Person("A"), genealogy.Person("A")),
            (genealogy.Person("A", ("B", "B")), genealogy.Person("B")),
        )
        for graph in invalid:
            with self.assertRaises(ValueError):
                genealogy.validate(graph)

    def test_shortest_generation_and_unchanged(self) -> None:
        family = (
            genealogy.Person("A"),
            genealogy.Person("B", ("A",)),
            genealogy.Person("C", ("A", "B")),
        )
        self.assertEqual(genealogy.ancestors(family, "C", 3), ((1, ("A", "B")),))
        self.assertEqual(family[2].parents, ("A", "B"))


class AvlTests(unittest.TestCase):
    """Все четыре типа поворотов и эксперимент на 10 000 ключей."""

    def test_four_rotations(self) -> None:
        for keys in ((3, 2, 1), (1, 2, 3), (3, 1, 2), (1, 3, 2)):
            tree = reduce(avl.insert, keys, None)
            self.assertEqual(tree.key, 2)
            self.assertEqual(list(avl.inorder(tree)), [1, 2, 3])
            self.assertTrue(avl.is_avl(tree))

    def test_10000_ascending(self) -> None:
        tree = reduce(avl.insert, range(10000), None)
        self.assertEqual(avl.height(tree), 14)
        self.assertTrue(avl.is_avl(tree))
        self.assertEqual(list(avl.inorder(tree)), list(range(10000)))

    def test_random_versions_and_sharing(self) -> None:
        keys = list(range(200))
        Random(306).shuffle(keys)
        tree = None
        for i, key in enumerate(keys):
            old = tree
            old_values = list(avl.inorder(old))
            tree = avl.insert(tree, key)
            self.assertTrue(avl.is_avl(tree))
            self.assertEqual(list(avl.inorder(tree)), sorted(keys[: i + 1]))
            self.assertEqual(list(avl.inorder(old)), old_values)
        self.assertIs(avl.insert(tree, keys[0]), tree)
        old_ids = avl.node_ids(tree)
        new_tree = avl.insert(tree, 201)
        new_ids = avl.node_ids(new_tree)
        self.assertLessEqual(len(new_ids - old_ids), 3 * avl.height(tree) + 1)
        self.assertGreater(len(new_ids & old_ids), 180)

    def test_invalid_rotations_and_invariant(self) -> None:
        with self.assertRaises(ValueError):
            avl.rotate_left(avl.Node(1))
        with self.assertRaises(ValueError):
            avl.rotate_right(avl.Node(1))
        self.assertFalse(avl.is_avl(avl.Node(1, height=7)))
        self.assertTrue(avl.is_avl(None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
