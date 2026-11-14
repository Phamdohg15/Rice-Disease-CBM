@echo off
cd /d "%~dp0"
call _env.bat
"%PY%" -u summarize.py
pause
