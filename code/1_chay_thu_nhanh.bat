@echo off
rem Chay thu 2 epoch (khong tinh vao ket qua), luu vao runs_smoke
cd /d "%~dp0"
call _env.bat
"%PY%" -u run.py --backbone resnet18 --mode cbm --seed 42 --epochs 2 --runs-root runs_smoke --overwrite
"%PY%" -u run.py --backbone resnet18 --mode baseline --seed 42 --epochs 2 --runs-root runs_smoke --overwrite
pause
