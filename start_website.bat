@echo off
cd /d "%~dp0"
echo Starting Certificate Reader... (first start takes about 20 seconds)
start "" http://127.0.0.1:8000
".venv\Scripts\python.exe" scripts\serve.py
pause
