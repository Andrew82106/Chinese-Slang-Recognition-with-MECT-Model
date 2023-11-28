@echo off

REM 检查是否传入参数
if "%~1"=="" (
    echo 请输入提交信息作为参数
    exit /b 1
)

set commit_message=%~1

REM 执行 Git 命令
git add .
git commit -m "%commit_message%"
git push origin master

echo 提交成功
