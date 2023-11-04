# Improved MECT Model Integration with Large-scale Language Models for Chinese Criminal Slang Recognition Framework Research


# usage

## 加载已有模型进行预测并详细显示结果：

将``--status``参数定义为 ``run`` 即可加载已有模型（模型路径需要改改）：

```py
python main.py --dataset weibo --status run --device cpu
```

## 原MECT4CNER模型训练命令：

- 微博数据集

```py
python main.py --dataset weibo
```