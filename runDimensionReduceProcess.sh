#!/bin/bash

if [ -z "$1" ]; then
    echo "请输入提交信息作为参数"
    exit 1
fi

eps="$1"
echo "$eps"

python cluster.py --mode clean_model_cache
python cluster.py --mode test_dimension_decline --eps="$eps"
python cluster.py --mode test

