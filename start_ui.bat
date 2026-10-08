@echo off
cd /d "%~dp0"
start "" http://127.0.0.1:8000
.venv\Scripts\python.exe app\server.py --port 8000
