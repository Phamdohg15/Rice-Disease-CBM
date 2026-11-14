@echo off
rem Chay thu nhanh transfer learning (2 epoch moi giai doan) -> runs_tl_smoke\
rem Kiem tra 3 truong hop: CBM hai giai doan, baseline hai giai doan, CBM tu dau.
cd /d "%~dp0"
call _env.bat
set ERR=0
"%PY%" -u run_tl.py --backbone resnet18 --mode cbm --seed 42 --strategy two_stage --epochs 2 --runs-root runs_tl_smoke --overwrite
if errorlevel 1 set ERR=1
"%PY%" -u run_tl.py --backbone resnet18 --mode baseline --seed 42 --strategy two_stage --epochs 2 --runs-root runs_tl_smoke --overwrite
if errorlevel 1 set ERR=1
"%PY%" -u run_tl.py --backbone resnet18 --mode cbm --seed 42 --strategy scratch --epochs 2 --runs-root runs_tl_smoke --overwrite
if errorlevel 1 set ERR=1
if "%ERR%"=="1" (echo CHAY THU CO LOI - xem error.txt trong runs_tl_smoke & pause & exit /b 1)
"%PY%" -u so_sanh_tl.py runs_tl_smoke
echo.
echo CHAY THU XONG. Neu khong co loi thi bam chay_tl.bat de chay that (sau khi chay_16c.bat da xong).
echo (File Excel luc nay chi la ban thu, se bi ghi de khi chay that.)
pause
