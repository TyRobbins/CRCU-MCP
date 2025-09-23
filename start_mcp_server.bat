@echo off
title Credit Union Analytics MCP Server
echo.
echo Credit Union Analytics MCP Server
echo ===================================
echo.
echo Python Path: C:\Python313\python.exe
echo Project Directory: c:\Users\Ty.Cinchy\Documents\GitHub\CRCU-MCP\credit_union_mcp
echo.
echo Starting server...
echo.

cd /d "c:\Users\Ty.Cinchy\Documents\GitHub\CRCU-MCP\credit_union_mcp"
"C:\Python313\python.exe" -m src.main

echo.
echo Server stopped. Press any key to exit...
pause > nul
