@echo off
rem Lap lai file Excel so sanh transfer learning tu cac luot da xong (khong train)
cd /d "%~dp0"
call _env.bat
"%PY%" -u so_sanh_tl.py
pause
