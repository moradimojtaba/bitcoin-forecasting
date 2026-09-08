@echo off
setlocal
cd /d "%~dp0"
python code\bitcoin_forecasting.py
if errorlevel 1 (
  echo.
  echo Forecasting program ended with an error. Check forecasting logs.
  pause
  exit /b 1
)
echo.
echo Forecasting run completed.
pause
