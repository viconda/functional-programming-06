"""Неизменяемое двоичное дерево поиска с разделением поддеревьев."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class Node:
    """Узел с уникальным целочисленным ключом."""

    key: int
    left: Tree = None
    right: Tree = None


Tree: TypeAlias = Node | None


def insert(tree: Tree, key: int) -> Node:
    """Рекурсивно вставить ключ, сохраняя старую версию дерева."""
    if tree is None:
        return Node(key)
    if key < tree.key:
        child = insert(tree.left, key)
        return tree if child is tree.left else Node(tree.key, child, tree.right)
    if key > tree.key:
        child = insert(tree.right, key)
        return tree if child is tree.right else Node(tree.key, tree.left, child)
    return tree


def insert_iterative(tree: Tree, key: int) -> Node:
    """Скопировать путь вставки без зависимости от лимита рекурсии."""
    path: list[tuple[Node, bool]] = []
    current = tree
    while current is not None:
        if key == current.key:
            assert tree is not None
            return tree
        go_left = key < current.key
        path.append((current, go_left))
        current = current.left if go_left else current.right
    result = Node(key)
    for parent, go_left in reversed(path):
        result = (
            Node(parent.key, result, parent.right)
            if go_left
            else Node(parent.key, parent.left, result)
        )
    return result


def minimum(tree: Node) -> int:
    """Рекурсивно найти минимальный ключ непустого дерева."""
    return tree.key if tree.left is None else minimum(tree.left)


def delete(tree: Tree, key: int) -> Tree:
    """Удалить ключ; узел с двумя потомками заменить преемником."""
    if tree is None:
        return None
    if key < tree.key:
        child = delete(tree.left, key)
        return tree if child is tree.left else Node(tree.key, child, tree.right)
    if key > tree.key:
        child = delete(tree.right, key)
        return tree if child is tree.right else Node(tree.key, tree.left, child)
    if tree.left is None:
        return tree.right
    if tree.right is None:
        return tree.left
    successor = minimum(tree.right)
    return Node(successor, tree.left, delete(tree.right, successor))


def inorder(tree: Tree) -> Iterator[int]:
    """Лениво обойти дерево по возрастанию, используя локальный стек."""
    stack: list[Node] = []
    current = tree
    while stack or current is not None:
        while current is not None:
            stack.append(current)
            current = current.left
        current = stack.pop()
        yield current.key
        current = current.right


def preorder(tree: Tree) -> Iterator[int]:
    """Рекурсивно обойти дерево в порядке корень, левое, правое."""
    if tree is not None:
        yield tree.key
        yield from preorder(tree.left)
        yield from preorder(tree.right)


def postorder(tree: Tree) -> Iterator[int]:
    """Рекурсивно обойти дерево в порядке левое, правое, корень."""
    if tree is not None:
        yield from postorder(tree.left)
        yield from postorder(tree.right)
        yield tree.key


def level_order(tree: Tree) -> Iterator[int]:
    """Обойти дерево по уровням с локальной очередью."""
    queue = deque([tree] if tree is not None else [])
    while queue:
        current = queue.popleft()
        yield current.key
        if current.left is not None:
            queue.append(current.left)
        if current.right is not None:
            queue.append(current.right)


def is_bst(tree: Tree) -> bool:
    """Проверить строгий порядок всех ключей, включая глубокие деревья."""
    previous: int | None = None
    for key in inorder(tree):
        if previous is not None and key <= previous:
            return False
        previous = key
    return True


def height(tree: Tree) -> int:
    """Число уровней: пустое дерево 0, лист 1; рекурсия не нужна."""
    stack = [(tree, 1)] if tree is not None else []
    maximum = 0
    while stack:
        node, depth = stack.pop()
        maximum = max(maximum, depth)
        if node.left is not None:
            stack.append((node.left, depth + 1))
        if node.right is not None:
            stack.append((node.right, depth + 1))
    return maximum


def from_sorted(keys: Sequence[int]) -> Tree:
    """Построить за O(n) дерево из строго возрастающей последовательности."""
    if any(keys[i] >= keys[i + 1] for i in range(len(keys) - 1)):
        raise ValueError("Ключи должны строго возрастать")

    def build(start: int, stop: int) -> Tree:
        if start >= stop:
            return None
        middle = (start + stop) // 2
        return Node(keys[middle], build(start, middle), build(middle + 1, stop))

    return build(0, len(keys))


def is_balanced(tree: Tree) -> bool:
    """За один проход проверить разность высот в каждом узле."""

    def check(node: Tree) -> int:
        if node is None:
            return 0
        left = check(node.left)
        if left < 0:
            return -1
        right = check(node.right)
        if right < 0 or abs(left - right) > 1:
            return -1
        return 1 + max(left, right)

    return check(tree) >= 0


def range_query(tree: Tree, lo: int, hi: int) -> Iterator[int]:
    """Выдать ключи включительного диапазона, отсекая лишние ветви."""
    if tree is None or lo > hi:
        return
    if lo < tree.key:
        yield from range_query(tree.left, lo, hi)
    if lo <= tree.key <= hi:
        yield tree.key
    if tree.key < hi:
        yield from range_query(tree.right, lo, hi)


def kth_smallest(tree: Tree, k: int) -> int:
    """Найти k-й ключ по возрастанию; нумерация начинается с 1."""
    if k < 1:
        raise IndexError("k должен быть положительным")
    for position, key in enumerate(inorder(tree), start=1):
        if position == k:
            return key
    raise IndexError("k превышает число ключей")
