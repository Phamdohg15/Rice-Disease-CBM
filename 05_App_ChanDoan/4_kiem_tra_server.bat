@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
"D:\Thesis\Project_Rice_Leaf_CNN\.venv\Scripts\python.exe" -X faulthandler kiem_tra_server.py > ket_qua_thu\kiem_tra_server_log.txt 2>&1
type ket_qua_thu\kiem_tra_server_log.txt
pause
