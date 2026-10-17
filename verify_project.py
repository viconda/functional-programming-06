"""Повторить тесты, проверить API и сохранить проверяемый итог."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Запустить unittest и проверить аннотации публичных функций ядра."""
    root = Path(__file__).resolve().parent
    output = root / "verification"
    output.mkdir(exist_ok=True)
    run = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=root,
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=dict(os.environ, PYTHONUTF8="1"),
        check=False,
    )
    transcript = run.stdout + run.stderr
    (output / "tests.txt").write_text(transcript, encoding="utf-8")
    if run.returncode:
        raise RuntimeError(transcript)
    count = int(re.search(r"Ran (\d+) tests", transcript).group(1))
    issues = []
    public = 0
    for folder in ("lab06", "lecture06"):
        for path in (root / folder).glob("*.py"):
            if path.name.startswith("demo_"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                    continue
                public += 1
                if not ast.get_docstring(node):
                    issues.append(f"{path.name}:{node.name}: no docstring")
                args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                if node.returns is None or any(a.annotation is None for a in args):
                    issues.append(f"{path.name}:{node.name}: missing annotation")
    summary = dict(tests=count, public_functions=public, api_issues=issues)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if issues:
        raise RuntimeError("\n".join(issues))
    print(f"Tests passed: {count}; documented and annotated public functions: {public}")


if __name__ == "__main__":
    main()
