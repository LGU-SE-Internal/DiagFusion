#!/bin/bash

# 创建diagfusion环境
uv venv diagfusion --python=3.10.16
source diagfusion/bin/activate
uv pip install -r requirements.txt
deactivate

# 创建dataprocess环境
uv venv dataprocess --python=3.13.4
source dataprocess/bin/activate
uv pip install -r ./DataProcess/requirements.txt
deactivate

echo "环境创建完成！"