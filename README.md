# Improved MECT Model Integration with Large-scale Language Models for Chinese Criminal Slang Recognition Framework Research

# usage

## stage1: 原MECT4CNER模型训练命令：

- 微博数据集

```py
python debug1.py --dataset weibo
```

## stage2: 加载已有模型进行预测并详细显示结果：

将``--status``参数定义为 ``run`` 即可加载已有模型（模型路径需要改改）：

```py
python debug1.py --dataset weibo --status run --device cpu
```

## stage3: 数据集输出为词向量：

```py
# 加载训练集对应测试集并转化为词向量
python debug1.py --dataset weibo --status generate --device cpu

# 加载外部数据集并转化为词向量
python debug1.py --dataset weibo --status generate --device cpu --extra_datasets tieba
```

## stage4: dbscan聚类分析


```py
python cluster.py
```