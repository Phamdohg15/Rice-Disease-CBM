@echo off
rem Lap lai file Excel so sanh 14 vs 16 concept tu cac luot da xong (khong train)
cd /d "%~dp0"
call _env.bat
"%PY%" -u so_sanh_14_16.py
pause
