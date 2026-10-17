"""Персистентная вставка в АВЛ-дерево с локальными поворотами."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class Node:
    """Неизменяемый узел с сохранённой высотой поддерева."""

    key: int
    left: Tree = None
    right: Tree = None
    height: int = 1


Tree: TypeAlias = Node | None


def height(tree: Tree) -> int:
    """Вернуть сохранённую высоту за O(1)."""
    return 0 if tree is None else tree.height


def _node(key: int, left: Tree, right: Tree) -> Node:
    return Node(key, left, right, 1 + max(height(left), height(right)))


def rotate_right(tree: Node) -> Node:
    """Создать версию после правого поворота без изменения старой."""
    pivot = tree.left
    if pivot is None:
        raise ValueError("Для правого поворота нужен левый потомок")
    return _node(pivot.key, pivot.left, _node(tree.key, pivot.right, tree.right))


def rotate_left(tree: Node) -> Node:
    """Создать версию после левого поворота без изменения старой."""
    pivot = tree.right
    if pivot is None:
        raise ValueError("Для левого поворота нужен правый потомок")
    return _node(pivot.key, _node(tree.key, tree.left, pivot.left), pivot.right)


def balance(tree: Node) -> Node:
    """Исправить нарушение АВЛ-инварианта одним или двумя поворотами."""
    difference = height(tree.left) - height(tree.right)
    if difference > 1:
        assert tree.left is not None
        if height(tree.left.left) < height(tree.left.right):
            tree = _node(tree.key, rotate_left(tree.left), tree.right)
        return rotate_right(tree)
    if difference < -1:
        assert tree.right is not None
        if height(tree.right.right) < height(tree.right.left):
            tree = _node(tree.key, tree.left, rotate_right(tree.right))
        return rotate_left(tree)
    return tree


def insert(tree: Tree, key: int) -> Node:
    """За O(log n) вставить ключ, разделяя неизменённые поддеревья."""
    if tree is None:
        return Node(key)
    if key < tree.key:
        child = insert(tree.left, key)
        if child is tree.left:
            return tree
        return balance(_node(tree.key, child, tree.right))
    if key > tree.key:
        child = insert(tree.right, key)
        if child is tree.right:
            return tree
        return balance(_node(tree.key, tree.left, child))
    return tree


def inorder(tree: Tree) -> Iterator[int]:
    """Рекурсивный ленивый обход ключей в порядке возрастания."""
    if tree is not None:
        yield from inorder(tree.left)
        yield tree.key
        yield from inorder(tree.right)


def is_avl(tree: Tree) -> bool:
    """Проверить порядок ключей, все высоты и баланс каждого узла."""

    def check(node: Tree, lo: int | None, hi: int | None) -> tuple[bool, int]:
        if node is None:
            return True, 0
        if (lo is not None and node.key <= lo) or (hi is not None and node.key >= hi):
            return False, 0
        left_ok, left_height = check(node.left, lo, node.key)
        right_ok, right_height = check(node.right, node.key, hi)
        expected = 1 + max(left_height, right_height)
        return (
            left_ok
            and right_ok
            and abs(left_height - right_height) <= 1
            and node.height == expected,
            expected,
        )

    return check(tree, None, None)[0]


def node_ids(tree: Tree) -> frozenset[int]:
    """Собрать идентичности узлов для проверки разделения памяти."""
    if tree is None:
        return frozenset()
    return frozenset((id(tree),)) | node_ids(tree.left) | node_ids(tree.right)
