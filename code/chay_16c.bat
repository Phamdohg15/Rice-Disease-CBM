@echo off
rem Ablation: CBM 16 concept (them C08, C13) cho 5 backbone x 3 seed = 15 luot -> runs_16c\
rem Luot nao xong roi thi tu bo qua (co the tat giua chung roi bam lai). Luot loi ghi vao runs_16c\loi.txt.
rem Cuoi cung tao so_sanh\SoSanh_14_vs_16_concept.xlsx
cd /d "%~dp0"
call _env.bat
if not exist runs_16c mkdir runs_16c
for %%B in (resnet18 densenet121 efficientnet_b0 convnext_tiny mobilenet_v2) do (
  for %%S in (42 43 44) do (
    "%PY%" -u run.py --backbone %%B --mode cbm --seed %%S --concept-set 16
    if errorlevel 1 echo %date% %time% LOI: %%B seed %%S>> runs_16c\loi.txt
  )
)
"%PY%" -u so_sanh_14_16.py
echo HOAN TAT 15 LUOT 16 CONCEPT. Xem so_sanh\SoSanh_14_vs_16_concept.xlsx
pause
