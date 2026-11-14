@echo off
rem Buoc 4: dong goi app thanh thu muc portable. CAN INTERNET, chay 1 lan, mat khoang 5-15 phut.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dong_goi\dong_goi.ps1"
pause
