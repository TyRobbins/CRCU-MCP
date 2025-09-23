@echo off
title Install MCP Dependencies
echo.
echo Credit Union Analytics MCP Dependencies
echo =======================================
echo.
echo Installing Python dependencies...
echo.

cd /d "C:\Users\Ty.Cinchy\Documents\GitHub\CRCU-MCP\credit_union_mcp"
"C:\Python313\python.exe" -m pip install --upgrade pip
"C:\Python313\python.exe" -m pip install -r requirements.txt

echo.
echo Dependencies installed. Press any key to exit...
pause > nul
