@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: 配置
set CONFIG=config\csi500_5min.yaml
set RUN_DIR=runs\5min
set RISK_FREE=0.02

echo ==========================================
echo  批量回测 — 5分钟多指标集模型
echo ==========================================
echo.

:: 遍历每个指标集子目录
for /d %%D in (%RUN_DIR%\*) do (
    set "iset=%%~nD"
    set "model_dir=%%D\models"
    set "data_file=%%D\dataset.parquet"
    set "report_dir=%%D\reports\manual"

    :: 查找 zip 模型文件（取第一个）
    set "model_zip="
    for %%Z in (!model_dir!\*.zip) do (
        if not defined model_zip set "model_zip=%%Z"
    )

    if not defined model_zip (
        echo [!iset!] 跳过：未找到模型文件
        echo.
        goto :next
    )

    if not exist "!data_file!" (
        echo [!iset!] 跳过：未找到数据文件 !data_file!
        echo.
        goto :next
    )

    echo [!iset!] 回测中 ...
    echo   模型: !model_zip!
    echo   数据: !data_file!
    echo   报告: !report_dir!

    finsys backtest ^
        --config !CONFIG! ^
        --model "!model_zip!" ^
        --data-file "!data_file!" ^
        --output "!report_dir!" ^
        --risk-free-rate !RISK_FREE!

    if errorlevel 1 (
        echo [!iset!] 回测失败
    ) else (
        echo [!iset!] 回测完成，报告已保存
    )
    echo.

    :next
)

echo ==========================================
echo  全部回测执行完毕
echo ==========================================
pause
