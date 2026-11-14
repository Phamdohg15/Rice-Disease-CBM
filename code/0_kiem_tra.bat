@echo off
cd /d "%~dp0"
call _env.bat
"%PY%" -u check.py
echo.
echo Xem ket qua trong check_report.txt
pause
