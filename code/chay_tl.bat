@echo off
rem Transfer learning: 5 backbone x 3 seed x (baseline, CBM) x (hai giai doan, tu dau) = 60 lenh -> runs_tl\
rem Lenh "hai giai doan" tao 2 luot: frozen (GD1) va two_stage (GD2). Tong cong 90 luot moi.
rem Luot nao xong roi thi tu bo qua: co the tat giua chung roi bam lai. Luot loi ghi vao runs_tl\loi.txt.
rem Cuoi cung tao so_sanh\SoSanh_TransferLearning.xlsx
cd /d "%~dp0"
call _env.bat
if not exist runs_tl mkdir runs_tl
for %%B in (resnet18 densenet121 efficientnet_b0 convnext_tiny mobilenet_v2) do (
  for %%S in (42 43 44) do (
    for %%M in (cbm baseline) do (
      for %%T in (two_stage scratch) do (
        "%PY%" -u run_tl.py --backbone %%B --mode %%M --seed %%S --strategy %%T
        if errorlevel 1 echo %date% %time% LOI: %%B %%M seed %%S %%T>> runs_tl\loi.txt
      )
    )
  )
  "%PY%" -u so_sanh_tl.py
)
echo HOAN TAT. Xem so_sanh\SoSanh_TransferLearning.xlsx
pause
