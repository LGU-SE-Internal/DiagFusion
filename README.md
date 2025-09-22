# DiagFusion: 多模态故障诊断与根因分析系统

DiagFusion是一个基于多模态数据融合的分布式系统故障诊断与根因定位算法实现。本项目是我们发表论文的配套代码。

## 🔧 主要特性

- **多模态数据融合**: 结合日志、指标、拓扑等多种数据源
- **深度学习模型**: 基于图神经网络和注意力机制的故障诊断
- **实时推理**: 支持在线故障检测和根因定位
- **标准化接口**: 兼容RCABench平台的标准算法接口

## 📁 项目结构

```
DiagFusion/
├── src/                    # 源代码目录
│   ├── diagfusion/        # 核心算法实现
│   ├── preprocess/        # 数据预处理
│   ├── exp/               # 实验相关代码
│   ├── utils/             # 工具函数
│   └── config/            # 配置文件
├── data/                  # 数据目录（需要挂载）
├── cache/                 # 缓存目录
├── main.py               # 主入口点
├── run.py                # 批量测试脚本
├── client.py             # 训练脚本
└── Dockerfile            # Docker镜像构建文件
```

## 🚀 快速开始

### 环境设置

```bash
# 安装依赖
uv sync

# 挂载数据集（需要根据实际环境调整）
sudo juicefs mount redis://10.10.10.119:6379/1 /mnt/jfs -d --cache-size=1024
mkdir -p data
ln -s /mnt/jfs/rcabench_dataset ./data/
```

### 环境变量配置

```bash
export CHECKPOINT_PATH=./data/middle/checkpoints/service_model.pt 
export DYNACONF_PATHS__METADATA=./data/middle/metadata 
export DYNACONF_PATHS__CKPT=./data/middle/checkpoints 
export RCABENCH_BASE_URL=http://10.10.10.220:32080
export RCABENCH_USERNAME=admin
export RCABENCH_PASSWORD=admin123
```

### 模型训练

```bash
sudo -E .venv/bin/python client.py --config rcabench.yaml
```

### 数据后处理

训练完成后需要复制相关文件：

```bash
sudo cp data/rcabench/demo/demo2/dgl/stratification_10/9/topology.pkl data/middle/metadata/ -f
sudo cp data/rcabench/demo/demo2/anomalies/service_instance_mapping.json data/middle/metadata/ -f
sudo rm -rf data/rcabench/demo/demo2
```

## 🔧 部署

### Docker构建

```bash
docker build -t 10.10.10.240/library/rca-algo-diagfusion:study .
```

### 算法上传

```bash
rca upload-algorithm-harbor ./
```

## 🧪 测试与评估

### 批量测试

```bash
sudo -E .venv/bin/python run.py batch-test --label 8.26run
```

### 指标评估

```bash
# 使用最新平台进行评估
rca cross-dataset-metrics -a diagfusion -d pair-diag -dv study-test --tag 8.26run
```


```
