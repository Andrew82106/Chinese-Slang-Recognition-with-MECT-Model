#!/bin/bash


if [ -z "$1" ]; then
    echo "请输入提交信息作为参数"
    exit 1
fi

commit_message="$1"

git add .
git commit -m "$commit_message"
git push origin master

echo "提交成功"
