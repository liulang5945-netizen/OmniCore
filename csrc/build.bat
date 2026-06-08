@echo off
REM ═══════════════════════════════════════════════════════════
REM  Taiji CUDA Engine — Windows 一键构建脚本
REM ═══════════════════════════════════════════════════════════
REM
REM  用法:
REM    cd csrc
REM    build.bat          (Release 模式)
REM    build.bat debug    (Debug 模式)
REM    build.bat clean    (清理构建目录)
REM

setlocal enabledelayedexpansion

set BUILD_DIR=%~dp0build
set CONFIG=Release

REM 参数解析
if "%1"=="debug" set CONFIG=Debug
if "%1"=="clean" goto :clean

echo.
echo ═══════════════════════════════════════════════════════════
echo   Taiji CUDA Engine - 构建 (%CONFIG%)
echo ═══════════════════════════════════════════════════════════
echo.

REM 检查依赖
where cmake >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cmake 未找到！请安装: pip install cmake
    exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python 未找到！
    exit /b 1
)

REM ═══════════════════════════════════════════════════════════
REM  MSVC 编译器自动检测与加载 (Torch.compile / Triton 必需)
REM ═══════════════════════════════════════════════════════════
where cl >nul 2>&1
if errorlevel 1 (
    echo [INFO] cl.exe 未在 PATH 中，尝试自动查找 MSVC...
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if not exist "!VSWHERE!" (
        echo [ERROR] vswhere.exe 未找到。请安装 Visual Studio Build Tools:
        echo        https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
        exit /b 1
    )
    for /f "delims=" %%i in ('"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "VSINSTALLDIR=%%i"
    if not defined VSINSTALLDIR (
        echo [ERROR] 未找到安装 MSVC 组件的 Visual Studio。请安装 "使用 C++ 的桌面开发" 工作负载。
        exit /b 1
    )
    call "!VSINSTALLDIR!\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] 加载 vcvars64.bat 失败: !VSINSTALLDIR!
        exit /b 1
    )
    where cl >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] 即使加载 vcvars64.bat 后仍找不到 cl.exe
        exit /b 1
    )
    echo [OK] MSVC 环境已自动加载: !VSINSTALLDIR!
) else (
    echo [OK] cl.exe 已在 PATH 中
)

REM 获取 PyTorch cmake 路径
for /f "delims=" %%i in ('python -c "import torch; print(torch.utils.cmake_prefix_path)"') do set TORCH_CMAKE=%%i
if "%TORCH_CMAKE%"=="" (
    echo [ERROR] PyTorch 未安装！请安装: pip install torch
    exit /b 1
)
echo [INFO] PyTorch CMake path: %TORCH_CMAKE%

REM 创建构建目录
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM CMake configure
echo.
echo [STEP 1/3] CMake Configure...
cd /d "%BUILD_DIR%"
cmake "%~dp0" ^
    -DCMAKE_PREFIX_PATH="%TORCH_CMAKE%" ^
    -DPYTHON_EXECUTABLE="%PYTHON%" ^
    -DCMAKE_BUILD_TYPE=%CONFIG%

if errorlevel 1 (
    echo [ERROR] CMake configure 失败！
    exit /b 1
)

REM CMake build
echo.
echo [STEP 2/3] CMake Build (%CONFIG%)...
cmake --build . --config %CONFIG% --parallel

if errorlevel 1 (
    echo [ERROR] CMake build 失败！
    exit /b 1
)

REM 复制产物到 python 目录
echo.
echo [STEP 3/3] 复制产物...
set OUTPUT_DIR=%~dp0python

REM 查找 .pyd 文件
for /r "%BUILD_DIR%\%CONFIG%" %%f in (*.pyd) do (
    copy "%%f" "%OUTPUT_DIR%\" >nul
    echo [OK] %%~nxf -^> %OUTPUT_DIR%\
)

REM 也检查 build 根目录
for /r "%BUILD_DIR%" %%f in (*.pyd) do (
    if not exist "%OUTPUT_DIR%\%%~nxf" (
        copy "%%f" "%OUTPUT_DIR%\" >nul
        echo [OK] %%~nxf -^> %OUTPUT_DIR%\
    )
)

echo.
echo ═══════════════════════════════════════════════════════════
echo   构建完成！
echo.
echo   验证: python -c "from csrc.python import TaijiEngine; print('OK')"
echo ═══════════════════════════════════════════════════════════
echo.

cd /d "%~dp0"
exit /b 0

:clean
echo [INFO] 清理构建目录: %BUILD_DIR%
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
echo [OK] 已清理
exit /b 0