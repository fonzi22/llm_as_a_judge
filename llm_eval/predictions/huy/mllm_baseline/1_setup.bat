@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

where python >nul 2>nul || (echo [X] Khong thay python trong PATH & pause & exit /b 1)

echo ===== [1/3] Cai / cap nhat Modal
python -m pip install -U modal || (echo [X] pip install loi & pause & exit /b 1)

echo ===== [2/3] Dang nhap Modal
if exist "%USERPROFILE%\.modal.toml" (
  echo Da co token Modal: %USERPROFILE%\.modal.toml
) else (
  python -m modal setup || (echo [X] modal setup loi & pause & exit /b 1)
)

echo ===== [3/3] Upload data len Volume hrvs-data (chi upload file thieu)
python prepare_data.py "%~dp0..\data" || (echo [X] Upload data loi - chay lai 1_setup.bat & pause & exit /b 1)

echo.
echo [OK] Setup xong. Buoc tiep: 2_smoke_test.bat
pause
