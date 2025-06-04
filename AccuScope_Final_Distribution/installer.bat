@echo off
echo --------------------------------------
echo Final GUI Installer
echo --------------------------------------

REM Check if Python is installed
where python >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Please install Python 3.9+ and try again.
    pause
    exit /b
)

REM Delete existing virtual environment if it exists
IF EXIST "venv\" (
    echo Removing existing virtual environment...
    rmdir /s /q venv
)

REM Create a new virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate and install dependencies
echo Activating and installing dependencies...
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

REM Run the main GUI
echo Running the Final GUI...
python Final_GUI\final_gui.py

pause
