@echo off
rem Tether launcher: tests / benchmarks / reports in one menu.
chcp 65001 >nul
title Tether Agent Workbench
cd /d "%~dp0"

:menu
cls
echo ================================================
echo   Tether Agent Workbench
echo ================================================
echo.
echo   [1] Run full test suite        (offline)
echo   [2] Run all offline benchmarks (5 experiments)
echo   [3] Run a single experiment
echo   [4] Run e2e experiment         (real LLM, needs DEEPSEEK_API_KEY)
echo   [5] Open summary report
echo   [6] Exit
echo.
set /p choice=Select an option (1-6): 

if "%choice%"=="1" goto tests
if "%choice%"=="2" goto all_bench
if "%choice%"=="3" goto single
if "%choice%"=="4" goto e2e
if "%choice%"=="5" goto report
if "%choice%"=="6" exit /b 0
goto menu

:tests
echo.
python -m pytest tests\ -q
echo.
pause
goto menu

:all_bench
echo.
python scripts\run_benchmark.py --all
echo.
pause
goto menu

:single
echo.
echo Experiments: compression ^| memory ^| drift ^| recovery ^| intercept ^| e2e
set /p exp=Experiment name: 
set /p n=Sample count (empty = all): 
if "%n%"=="" (
    python scripts\run_benchmark.py --experiment %exp%
) else (
    python scripts\run_benchmark.py --experiment %exp% --num-samples %n%
)
echo.
pause
goto menu

:e2e
echo.
set "key=%DEEPSEEK_API_KEY%"
if "%key%"=="" (
    echo [WARN] DEEPSEEK_API_KEY not set - will fall back to MockProvider.
    set /p key=Paste your DeepSeek API key (empty = mock mode): 
)
set /p n=Sample count (default 5): 
if "%n%"=="" set n=5
set DEEPSEEK_API_KEY=%key%
set TETHER_LLM_MODEL=deepseek-v4-flash
python scripts\run_benchmark.py --experiment e2e --num-samples %n%
set DEEPSEEK_API_KEY=
echo.
pause
goto menu

:report
echo.
start "" "src\tether\benchmarks\results\summary.md"
goto menu
