@echo off
rem Chay thu nhanh ban 16 concept: ResNet18, seed 42, 2 epoch -> runs_16c_smoke\
rem Muc dich: kiem tra code chay duoc truoc khi chay that 15 luot (khoang 3,5 gio).
cd /d "%~dp0"
call _env.bat
"%PY%" -u run.py --backbone resnet18 --mode cbm --seed 42 --concept-set 16 --epochs 2 --runs-root runs_16c_smoke --overwrite
if errorlevel 1 (echo CHAY THU BI LOI - xem runs_16c_smoke\resnet18_cbm_seed42\error.txt & pause & exit /b 1)
"%PY%" -u so_sanh_14_16.py runs_16c_smoke
echo.
echo CHAY THU XONG. Neu khong co loi thi bam chay_16c.bat de chay that.
echo (File Excel luc nay chi la ban thu, se bi ghi de khi chay that.)
pause
