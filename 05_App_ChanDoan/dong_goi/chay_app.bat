@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
title Chan doan benh la lua
echo Dang khoi dong app chan doan benh la lua (khoang 10-20 giay)...
echo Trinh duyet se tu mo. Dong cua so nay de tat app.
"%~dp0python\python.exe" server.py
pause
