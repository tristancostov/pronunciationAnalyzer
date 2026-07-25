@echo off
setlocal
cd /d "%~dp0"
set "PATH=%CD%\cuda-runtime;%PATH%"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
"%CD%\Python\Python39\python.exe" "%CD%\gui_app.py"
if errorlevel 1 pause
