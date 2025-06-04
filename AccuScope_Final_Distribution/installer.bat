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

REM Check if venv exists
IF NOT EXIST "venv\" (
    echo Creating virtual environment...
    python -m venv venv

    echo Activating and installing dependencies...
    call venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt
) ELSE (
    echo Virtual environment already exists.
    echo Activating environment...
    call venv\Scripts\activate
)

REM Run the main GUI
echo Running the Final GUI...
python Final_GUI\final_gui.py

pause
