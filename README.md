# Improved MECT Model Integration with Large-scale Language Models for Chinese Criminal Slang Recognition Framework Research

# usage

## stage1: 原MECT4CNER模型训练命令：

- 微博数据集

```shell
python main.py --dataset weibo
```

## stage2: 加载已有模型进行预测并详细显示结果：

将``--status``参数定义为 ``run`` 即可加载已有模型（模型路径需要改改）：

```shell
python main.py --dataset weibo --status run --device cpu
```

## stage3: 数据集输出为词向量：

```shell
# 加载训练集对应测试集并转化为词向量
python main.py --dataset weibo --status generate --device cpu

# 加载外部数据集并转化为词向量
python main.py --dataset weibo --status generate --device cpu --extra_datasets tieba

# 加载外部数据集并转化为词向量（重建wiki.pkl，重建的时候必须有test.pkl存在，并且重建之前需要先清除wiki数据集的缓存）
python cluster.py --mode clean_model_cache
python main.py --dataset msra --status generate --device cpu --extra_datasets wiki

# 加载外部数据集并转化为词向量（test.pkl）
python cluster.py --mode refresh_test
```

## stage4: 生成文本解析结果

```shell
# 使用非降维算法进行测试
python cluster.py --mode generate

# 使用降维算法进行测试
python cluster.py --mode test_dimension_decline
```

## stage5: 调用Fastnlp接口评测解析结果

```shell
python cluster.py --mode test
```

## 快速操作

- 对wiki数据集进行向量化+生成文本聚类分析结果+测试结果
```shell
bash RunCluster.sh
```

- 删除缓存+使用降维算法进行测试+测试结果
```shell
bash runDimensionReduceProcess.sh 0.8
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
- 运行wiki数据集增强脚本
```shell
python cluster.py --mode expandBaseData
python cluster.py --mode ConvertExpandedData
```
- 函数缓存清除
```shell
python cluster.py --mode clean_function_cache
```
- 数据集缓存清除
```shell
python cluster.py --mode clean_model_cache
```
- 合并wiki和大模型增强数据
```shell
python cluster.py --mode strengthen_wiki
```