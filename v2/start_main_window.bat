@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "ENV_NAME=yolo"
set "CONDA_ACTIVATED="

echo 正在激活虚拟环境 "%ENV_NAME%"...

for %%C in ("%USERPROFILE%\anaconda3" ^
            "%USERPROFILE%\miniconda3" ^
            "C:\ProgramData\anaconda3" ^
            "C:\ProgramData\miniconda3") do (
    if not defined CONDA_ACTIVATED (
        if exist "%%~C\Scripts\activate.bat" (
            call "%%~C\Scripts\activate.bat" "%ENV_NAME%"
            if not errorlevel 1 (
                set "CONDA_ACTIVATED=1"
            )
        )
    )
)

if not defined CONDA_ACTIVATED (
    echo 未找到可用的 conda 激活脚本，请手动修改 ^"start_main_window.bat^" 以匹配您的 conda 安装路径。
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"

echo 正在启动 main_window.py...
python main_window.py

if errorlevel 1 (
    echo 程序执行过程中出现错误。
) else (
    echo 程序已正常退出。
)

pause
endlocal

