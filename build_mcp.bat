@echo off
title Credit Union MCP Build
echo.
echo Credit Union Analytics MCP Build
echo =================================
echo.
echo Building standalone executable...
echo.

cd /d "c:\Users\Ty.Cinchy\Documents\GitHub\CRCU-MCP\credit_union_mcp"
"C:\Python313\python.exe" build.py

echo.
echo Build complete. Press any key to exit...
pause > nul
