@echo off
rem Chay 5 cap = 30 luot, theo thu tu bang so sanh. Co the tat giua chung roi bam lai: luot da xong se bo qua.
cd /d "%~dp0"
for %%B in (resnet18 densenet121 efficientnet_b0 convnext_tiny mobilenet_v2) do call chay_cap.bat %%B
call _env.bat
"%PY%" -u summarize.py
echo HOAN TAT 30 LUOT. Xem runs\BANG_SO_SANH.md
pause
