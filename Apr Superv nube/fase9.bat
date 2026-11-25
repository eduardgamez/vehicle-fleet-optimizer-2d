@echo off
rem Lanza o reanuda la fase 9 (ver fase9_lanzar.ps1).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0fase9_lanzar.ps1"
timeout /t 5 >nul
