@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (set "PYTHON_COMMAND=python") else (set "PYTHON_COMMAND=py -3")
%PYTHON_COMMAND% -m unittest discover -s tests -v
if errorlevel 1 goto end
%PYTHON_COMMAND% demo_lecture.py
%PYTHON_COMMAND% demo_algorithms.py
%PYTHON_COMMAND% demo_trees.py
:end
pause
