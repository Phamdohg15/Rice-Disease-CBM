@echo off
rem Buoc 3: chay thu app web bang Python cua may ca nhan (truoc khi dong goi)
chcp 65001 >nul
cd /d "%~dp0"
set "PY=D:\Thesis\Project_Rice_Leaf_CNN\.venv\Scripts\python.exe"
set PYTHONIOENCODING=utf-8
"%PY%" -c "import fastapi, uvicorn" 2>nul
if errorlevel 1 (
  echo Chua co FastAPI/uvicorn trong moi truong Python nay - dang cai, can internet...
  "%PY%" -m pip install fastapi uvicorn
)
"%PY%" server.py
pause
