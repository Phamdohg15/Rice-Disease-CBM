@echo off
rem Chay 1 cap: chay_cap.bat resnet18   (6 luot: CBM va baseline x 3 seed)
rem Luot nao xong roi thi tu bo qua; luot loi ghi vao runs\loi.txt va chay tiep
cd /d "%~dp0"
call _env.bat
if "%~1"=="" (echo Thieu ten backbone & exit /b 1)
if not exist runs mkdir runs
for %%S in (42 43 44) do (
  for %%M in (cbm baseline) do (
    "%PY%" -u run.py --backbone %~1 --mode %%M --seed %%S
    if errorlevel 1 echo %date% %time% LOI: %~1 %%M seed %%S>> runs\loi.txt
  )
)
"%PY%" -u summarize.py
