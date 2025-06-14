@echo off
:: Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

:: Clean previous builds
rmdir /s /q build dist
if exist app.spec del app.spec

:: Build standalone EXE with model included
pyinstaller --onefile --noconsole --add-data "inswapper_128.onnx;." app.py

echo Build complete! Find app.exe in dist\ folder.
pause
