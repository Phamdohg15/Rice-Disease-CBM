# Dong goi app chan doan thanh thu muc portable (chay bang CPU, khong can cai gi tren may khac).
# Chay 1 lan tren may ca nhan (CAN INTERNET): bam dup 5_dong_goi.bat trong D:\Thesis\App
# Ket qua: D:\Thesis\App_DongGoi\ChanDoanLua  -> chep ca thu muc nay sang may khac, bam chay_app.bat

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$src  = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)   # D:\Thesis\App
$dist = Join-Path (Split-Path -Parent $src) "App_DongGoi\ChanDoanLua"
$py   = Join-Path $dist "python"
$tmp  = Join-Path $env:TEMP "chandoanlua_build"

function Step($t) { Write-Host ""; Write-Host "=== $t ===" -ForegroundColor Green }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "Loi o buoc: $what (ma loi $LASTEXITCODE)" } }

Step "1/6 Tao thu muc $dist"
if (Test-Path $dist) { Write-Host "Da co ban cu -> xoa va tao lai"; Remove-Item -Recurse -Force $dist }
New-Item -ItemType Directory -Force $dist, $tmp | Out-Null

Step "2/6 Chep file cua app (chi phan can de chay)"
foreach ($i in "server.py", "app_config.json", "core", "model_src", "model", "knowledge", "static") {
    Copy-Item -Recurse (Join-Path $src $i) $dist
}
foreach ($i in "chay_app.bat", "kiem_tra_nhanh.py", "HUONG_DAN.txt") {
    Copy-Item (Join-Path $src "dong_goi\$i") $dist
}
Get-ChildItem $dist -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
# 5 anh mau (1 anh/lop, lay tu tap test) de demo va de tu kiem tra
$anh = Join-Path $dist "anh_mau"
New-Item -ItemType Directory -Force $anh | Out-Null
$T = "D:\Thesis\Dataset\Data\images\test"
$mau = @{
  "01_dao_on.jpg"  = "$T\01_dao_on\img_8201c1374ec3444cf1afba4f06dae6603aefa887f1e273b0682bbc1714e5f6a7.jpg";
  "02_dom_nau.jpg" = "$T\02_dom_nau\img_840813353b03eae5" ;
  "03_bac_la.jpg"  = "$T\03_bac_la\img_28eba68c3fd4484bb1e8859aaf9b78aaa5732b0c62946104056c2f47f448dd93.jpg";
  "04_kho_van.jpg" = "$T\04_kho_van\img_2c6d8e71187a1827c008104c43cfc5f351038f78626e977274863bc839882f7f.jpg";
  "05_la_khoe.jpg" = "$T\05_la_khoe\img_fc68e2370dec6d642c7999fe9fef493805d50f3f908ca73b2e9824f9e2d3b1c3.jpg"
}
foreach ($k in $mau.Keys) {
    $f = $mau[$k]
    if (-not (Test-Path $f)) { $f = (Get-ChildItem "$f*" | Select-Object -First 1).FullName }  # anh dom nau: tim theo dau ten
    Copy-Item $f (Join-Path $anh $k)
}

Step "3/6 Tai Python 3.11 ban nhung (embeddable)"
$zip = Join-Path $tmp "python-3.11.9-embed-amd64.zip"
if (-not (Test-Path $zip)) { Invoke-WebRequest "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip" -OutFile $zip }
Expand-Archive $zip $py -Force
# cho phep Python ban nhung dung site-packages (de cai thu vien)
Set-Content (Join-Path $py "python311._pth") "python311.zip`r`n.`r`nLib\site-packages`r`nimport site" -Encoding ascii

Step "4/6 Cai pip"
$getpip = Join-Path $tmp "get-pip.py"
if (-not (Test-Path $getpip)) { Invoke-WebRequest "https://bootstrap.pypa.io/get-pip.py" -OutFile $getpip }
& "$py\python.exe" $getpip --no-warn-script-location; Check "cai pip"

Step "5/6 Cai thu vien (PyTorch ban CPU ~200 MB, FastAPI...) - mat vai phut"
& "$py\python.exe" -m pip install --no-warn-script-location --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu; Check "cai torch"
& "$py\python.exe" -m pip install --no-warn-script-location --no-cache-dir fastapi uvicorn pillow numpy; Check "cai fastapi"
# Thu vien chay C++ cua Windows: chep kem de may khac khong can cai Visual C++ Redistributable
foreach ($d in "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll", "vcomp140.dll") {
    $s = Join-Path "$env:WINDIR\System32" $d
    if ((Test-Path $s) -and -not (Test-Path (Join-Path $py $d))) { Copy-Item $s $py }
}

Step "6/6 Tu kiem tra ban dong goi (chay bang Python ben trong thu muc, CPU)"
& "$py\python.exe" (Join-Path $dist "kiem_tra_nhanh.py"); Check "tu kiem tra"

$mb = [math]::Round(((Get-ChildItem $dist -Recurse -File | Measure-Object Length -Sum).Sum) / 1MB)
Write-Host ""
Write-Host "XONG. Thu muc dong goi: $dist  ($mb MB)" -ForegroundColor Green
Write-Host "Chep ca thu muc ChanDoanLua sang may khac (USB...), bam dup chay_app.bat."
