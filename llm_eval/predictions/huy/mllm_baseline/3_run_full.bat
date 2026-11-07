@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

rem Mo 2 cua so chay SONG SONG (moi model 1 GPU rieng tren Modal) cho nhanh.
rem Giu may bat + co mang toi khi xong. Bi dut -> chay lai file nay, script tu resume.
rem Chi chay 1 model:  3_run_full.bat qwen3-vl-8b   hoac   3_run_full.bat llava-ov-7b

if not "%~1"=="" (
  python -m modal run modal_qwen_vl.py --model %~1
  python show_results.py
  pause
  exit /b
)

start "FULL qwen3-vl-8b" cmd /k "chcp 65001 >nul & cd /d %~dp0 & python -m modal run modal_qwen_vl.py --model qwen3-vl-8b"
start "FULL llava-ov-7b" cmd /k "chcp 65001 >nul & cd /d %~dp0 & python -m modal run modal_qwen_vl.py --model llava-ov-7b"
echo Da mo 2 cua so. Khi ca 2 in dong "OK: results\..." thi chay 4_show_results.bat
pause
