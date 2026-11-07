@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

echo ===== Smoke test Qwen3-VL-8B (10 cau) - lan dau ~10-15 phut (build image + tai weights)
python -m modal run modal_qwen_vl.py --model qwen3-vl-8b --limit 10
if errorlevel 1 echo [X] qwen3-vl-8b LOI - gui log cho Claude

echo.
echo ===== Smoke test LLaVA-OV-7B (10 cau)
python -m modal run modal_qwen_vl.py --model llava-ov-7b --limit 10
if errorlevel 1 echo [X] llava-ov-7b LOI - gui log cho Claude

echo.
echo ===== Ket qua
python show_results.py limit10
echo.
echo Neu ca 2 model: rows=10, error=0, truncated=0 va PRED hop ly -^> chay 3_run_full.bat
pause
