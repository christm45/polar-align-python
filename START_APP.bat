@echo off
echo ========================================
echo  POLAR ALIGNMENT APP - STARTER
echo ========================================
echo.
echo Stopping any running instances...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq polar*" 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Starting Polar Alignment App...
echo.
echo The app will start in a moment...
echo Once you see "Running on http://127.0.0.1:5000", you can:
echo   1. Open your browser to: http://localhost:5000
echo   2. Press Ctrl+F5 to hard refresh the page
echo.
echo ========================================
echo.

python polar_align.py

pause

