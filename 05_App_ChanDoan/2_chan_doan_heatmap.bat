@echo off
rem Chan doan heatmap: ve nhieu cach giai thich cho 11 anh test (5 lop + 6 anh dom nau)
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
"D:\Thesis\Project_Rice_Leaf_CNN\.venv\Scripts\python.exe" chan_doan_heatmap.py > ket_qua_thu\chan_doan_log.txt 2>&1
type ket_qua_thu\chan_doan_log.txt
pause
