# Improved MECT Model Integration with Large-scale Language Models for Chinese Criminal Slang Recognition Framework Research

# usage

## stage1: 原MECT4CNER模型训练命令：

- 微博数据集

```py
python main.py --dataset weibo
```

## stage2: 加载已有模型进行预测并详细显示结果：

将``--status``参数定义为 ``run`` 即可加载已有模型（模型路径需要改改）：

```py
python main.py --dataset weibo --status run --device cpu
```

## stage3: 数据集输出为词向量：

```py
# 加载训练集对应测试集并转化为词向量
python main.py --dataset weibo --status generate --device cpu

# 加载外部数据集并转化为词向量
python main.py --dataset weibo --status generate --device cpu --extra_datasets tieba
python main.py --dataset msra --status generate --device cpu --extra_datasets wiki
```

## stage4: 生成文本解析结果

```py
python cluster.py --mode generate
```

## stage5: 评测解析结果

```py
python cluster.py --mode test
```

## 快速操作

- 对wiki数据集进行向量化+生成文本聚类分析结果+测试结果
```shell
bash RunCluster.sh
```

## 其他操作

- 运行降维算法比较测试脚本
```shell
python cluster.py --mode lowDimensionLab
```
- 运行暗语词汇词向量降维效果展示脚本
```shell
python cluster.py --mode CompareSensitiveWordLab
```