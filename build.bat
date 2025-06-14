@echo off
pip install -r requirements.txt
pip install pyinstaller

:: Optional: remove previous build
rmdir /s /q build dist
del app.spec

:: Create executable with PyInstaller
pyinstaller --onefile --noconsole app.py

echo.
echo ✅ DONE! Your EXE is in the dist/ folder.
pause
