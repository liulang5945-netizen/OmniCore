@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
title OmniCore 热更新部署

echo ============================================
echo   OmniCore 热更新部署工具
echo ============================================
echo.

set EXE_DIR=dist\OmniCore

:: ====== 参数处理 ======
if "%~1"=="" goto :full_update
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help
if "%~1"=="--watch" goto :watch_mode
if "%~1"=="-w" goto :watch_mode
if "%~1"=="--list" goto :list_patches
if "%~1"=="-l" goto :list_patches
if "%~1"=="--file" goto :deploy_file
if "%~1"=="-f" goto :deploy_file

echo [ERROR] 未知参数: %~1
echo 用法: hotupdate.bat [--watch^|--file^|--list^|--help]
goto :end

:: ====== 完整热更新（默认模式） ======
:full_update
echo [1] 编译前端...
cd frontend
call npm run build
if errorlevel 1 (
    echo [ERROR] 前端编译失败！
    cd ..
    goto :end
)
cd ..

echo [2] 部署前端更新...
if not exist "%EXE_DIR%\update_frontend\assets" mkdir "%EXE_DIR%\update_frontend\assets" 2>nul
copy /Y "frontend\dist\index.html" "%EXE_DIR%\update_frontend\index.html" >nul
xcopy /Y /Q "frontend\dist\assets\*.js"  "%EXE_DIR%\update_frontend\assets\" >nul
xcopy /Y /Q "frontend\dist\assets\*.css" "%EXE_DIR%\update_frontend\assets\" >nul
echo    OK

echo.
echo ============================================
echo   完成！按 F5 刷新页面生效
echo ============================================
goto :end

:: ====== 文件监听模式 ======
:watch_mode
echo 启动文件监听模式...
echo 修改 .py 文件 -> 自动部署到 update_code/
echo 修改前端文件 -> 自动重编译前端
echo.
python build_scripts/hot_update.py --watch
goto :end

:: ====== 部署单个文件 ======
:deploy_file
shift
if "%~1"=="" (
    echo [ERROR] 请指定要部署的文件路径
    echo 用法: hotupdate.bat --file api/routes_chat.py
    goto :end
)
echo 部署文件: %~1
python build_scripts/hot_update.py --file %~1
goto :end

:: ====== 列出已安装补丁 ======
:list_patches
python build_scripts/hot_update.py --list
goto :end

:: ====== 帮助信息 ======
:show_help
echo.
echo 用法:
echo   hotupdate.bat              完整前端编译+部署（默认）
echo   hotupdate.bat --watch      文件监听模式（自动检测变更）
echo   hotupdate.bat --file FILE  部署单个 Python 补丁文件
echo   hotupdate.bat --list       列出已安装的补丁
echo   hotupdate.bat --help       显示此帮助信息
echo.
echo 示例:
echo   hotupdate.bat --file api/routes_chat.py
echo   hotupdate.bat --file model/trainer.py
echo   hotupdate.bat --watch
echo.
goto :end

:end
pause