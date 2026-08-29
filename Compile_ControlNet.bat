@echo off
setlocal
cd /d "%~dp0"
call venv\Scripts\activate.bat

echo ========================================================
echo ControlNet Compiler (TensorRT)
echo ========================================================
echo Select which ControlNet you want to compile:
echo 1. Depth (MiDaS) - Preserves 3D volume and hand shape
echo 2. Canny Edge - Preserves 2D texture and facial features
echo 3. Lineart - Turns webcam into a sketch-based drawing
echo 4. OpenPose - Skeletal tracking (ignores clothing/body shape)
echo 5. Multi (Depth + Canny) - [WARNING: EXPERIMENTAL, High VRAM]
echo ========================================================
set /p mode="Enter number (1-5): "

if "%mode%"=="1" set ctype=depth
if "%mode%"=="2" set ctype=canny
if "%mode%"=="3" set ctype=lineart
if "%mode%"=="4" set ctype=openpose
if "%mode%"=="5" set ctype=multi

if not defined ctype (
    echo Invalid selection.
    pause
    exit /b
)

echo Compiling %ctype% engine for TensorRT...
echo This will take 10-15 minutes and will consume up to 10GB of VRAM.
echo Do not play games or run heavy GPU tasks during this process.
echo.
python compile_controlnet_fused.py --type %ctype%

echo.
echo Compilation finished! You can now use %ctype% mode in the launcher.
pause
