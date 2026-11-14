@echo off
rem Buoc 1: kiem tra loi du doan cua app (dung Python cua may ca nhan). Ket qua ghi vao ket_qua_thu\buoc1_log.txt
chcp 65001 >nul
cd /d "%~dp0"
set "PY=D:\Thesis\Project_Rice_Leaf_CNN\.venv\Scripts\python.exe"
set PYTHONIOENCODING=utf-8
if not exist ket_qua_thu mkdir ket_qua_thu
if not exist model\best.pt copy /y "D:\Thesis\CBM_V18\runs\convnext_tiny_cbm_seed43\best.pt" model\best.pt >nul
(
echo ===== 1. Du doan 5 anh test, 1 anh moi lop (GPU neu co) =====
"%PY%" du_doan_1_anh.py "D:\Thesis\Dataset\Data\images\test\01_dao_on\img_8201c1374ec3444cf1afba4f06dae6603aefa887f1e273b0682bbc1714e5f6a7.jpg" "D:\Thesis\Dataset\Data\images\test\02_dom_nau\img_43d977b2c7a22534ba6bbb843374806b60d8a9b34d03e3cd8a7d03fb184be509.jpg" "D:\Thesis\Dataset\Data\images\test\03_bac_la\img_28eba68c3fd4484bb1e8859aaf9b78aaa5732b0c62946104056c2f47f448dd93.jpg" "D:\Thesis\Dataset\Data\images\test\04_kho_van\img_2c6d8e71187a1827c008104c43cfc5f351038f78626e977274863bc839882f7f.jpg" "D:\Thesis\Dataset\Data\images\test\05_la_khoe\img_fc68e2370dec6d642c7999fe9fef493805d50f3f908ca73b2e9824f9e2d3b1c3.jpg"
echo.
echo ===== 2. So khop app voi ket qua danh gia tren 126 anh test (GPU) =====
"%PY%" kiem_tra_khop.py
echo.
echo ===== 3. Nhu muc 2 nhung ep chay CPU, de do toc do tren may khong co GPU =====
"%PY%" kiem_tra_khop.py --cpu
) > ket_qua_thu\buoc1_log.txt 2>&1
type ket_qua_thu\buoc1_log.txt
echo.
echo Xong. Ket qua: ket_qua_thu\buoc1_log.txt va cac anh *_heatmap.png
pause
