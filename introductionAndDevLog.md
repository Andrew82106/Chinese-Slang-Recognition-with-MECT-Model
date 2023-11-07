# 基于MECT4CNER修改的版本

原仓库：[MECT4CNER_Repo](https://github.com/CoderMusou/MECT4CNER)

## dev log

- 2023.8.23 开始尝试对英文词汇添加词根拆分功能。运行中英混搭数据集命令：

```bash
python main.py --dataset demo
```

- 2023.9.3 调试复盘

思考了一下，发现之前处理英文单词的思路不太对。

之前的思路是在CNN偏旁embedding中对英文单词去找词根，达到和偏旁部首类似的效果

但是在整个模型中lattice embedding部分模型也会把英文单词拆开，所以上面单独找词根其实是无用功

而且单独找词根还会出现句子长度在lattice embedding和radical embedding中不一致的情况，一边是把单词直接拆成字母，另一边是把单词当做一个汉字

所以正确的做法应该是将单词拆成字母直接放进去

所以应该一开始在处理数据集的时候就把单词拆开

哎搞了这么久原来是这个的原因，有点气的

不过这个自己做的数据集总是会报错说``Invalid instance which ends at line:xxx has been dropped``

找了找，应该是数据集的问题，数据集中有爬取的时候没处理好的nbsp符号，把数据集处理一下就好了

- 2023.9.3 中英混搭数据集适配修改总结

现在看来上面的思路没有问题，就是直接将英文一个字母一个字母的拆开放到数据集中去，其他的其实不用修改，硬要说的话也就改AdaptSelfAttention.py里面那个句子长度（原仓库issue里面提到过的）

不过现在遇到的问题是，英文加入后，句子长度会急剧增加，句子中的可能词组数量（span数量）也会急剧增加，这样就使得模型的AdaptSelfAttention.py里面的那一部分的维度数增多了，这样的话24G显存就不太够用。

一个解决方法是使用多卡训练，大一点的GPU训练等等。多卡训练还没研究明白，大的GPU没有。

另一个解决办法是将数据集中长度太长的句子直接删掉，现在就是这样做的，训练集已经跑起来了。现在是凌晨，等今天早上起床看看效果如何。



# 11.5 日工作工作总结


## MECT4CNER 模型结构展示

重新梳理了model.py文件中的MECT4CNER模型结构，具体结构如下：

![WechatIMG1032.jpg](md_cache/WechatIMG1032.jpg)

## 完善了CSR_MECTNER模型

基于MECT4CNER模型的结构，新写了一个CSR_MECTNER模型，将这个模型的Output改成了输出中间结果：

```py
if self.training:
            loss = self.crf(pred, target, mask).mean(dim=0)
            return {'loss': loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, mask)
            result = {
                'pred': pred, 
                'fusion': fusion, 
                'radical_encoded': radical_encoded, 
                'char_encoded': char_encoded,
                'seq_output': seq_output,
                'batch_size': batch_size,
                'max_seq_len_and_lex_num': max_seq_len_and_lex_num,
                'max_seq_len': max_seq_len
                
            }

            return result
```

接下来应该就能进行下一步词向量的合成了

# 11.6 日工作总结

 - 完成了CSR_MECTNER模型的结构调整，将输出定向到了seq_output变量中，研究了seq_output变量的形状
 
 - 确定了接下来的思路：
 
 首先使用任意方法A进行词向量合成，然后使用dbscan进行聚类，观察不同的数据集对于同一个词语效果是否有区别，以此来判断算法效果
 
 然后结合效果，对方法A和聚类算法进行修改优化
 
 # 11.7 日工作总结
 
 - 构建了字向量到词向量的模块``CharacterToWord.py``，后续可以将提升后的模型接入该类中
 
 - 尝试构建贴吧数据集
 
 这里需要注意的一点是，贴吧数据集不需要做标记，也就是所有的字的标记都是O。这样的话直接用这个数据集套上已有的模型，就可以直接输出词向量，可以认为这个词向量是基于已有的MECT4CNER模型的拟合的模型构建的。
 
 但是仅仅替换数据集好像不太行，因为fastnlp的Trainer训练的时候存的是模型的statedict，因此加载模型的时候需要把原模型初始化出来后再load state dict，否则的话模型size不匹配，应该得想个别的办法弄弄
 
 尝试用微博训练出来的模型去跑贴吧的数据集，然后发现报这个错：

 
 ```py
 TypeError: forward() missing 4 required positional arguments: 'bigrams', 'seq_len', 'lex_num', and 'target'
 ```
 
 一开始以为是表的问题，试了试原数据集，发现又能跑动，然后就估计是原数据表和新数据表的内容不一样
 
 但是翻来覆去看，就没看到啥问题，两个表的表头和数据类型都是一模一样的
 
 然后以为是词表太大了，把词表搞小一点，贴吧数据集只拿了11句话来弄，但是小词表也是不行
 
 很无语
 
 到处找了一下午，最后突然灵光乍现想起了这个：
 
 > fastnlp.core.Trainer的forward的参数来自于datasets变量中被设为input的field
 
 然后调试了一下贴吧数据集的表的信息：
 

| field_names | chars | target | bigrams | seq_len | lexicons | raw_chars | lex_num | lex_s | lex_e | lattice | pos_s | pos_e |
|-------------|-------|--------|---------|---------|----------|-----------|---------|-------|-------|---------|-------|-------|
|   is_input  | False | False  |  False  |  False  |  False   |   False   |  False  | False | False |   True  |  True |  True |
|  is_target  | False | False  |  False  |  False  |  False   |   False   |  False  | False | False |  False  | False | False |
| ignore_type |       |        |         |         |          |           |         |       |       |  False  | False | False |
|  pad_value  |       |        |         |         |          |           |         |       |       |    0    |   0   |   0   |


 然后发现，上面报错的那四个变量，确实没有放到input里面去。。。。。。一下午时间就干了个这个。。。。。。。
 
 同时，如果外接数据集过大的话，确实会出现词表溢出的情况，比如下面：
 ```py
 Traceback (most recent call last):
  File "/root/autodl-tmp/Chinese-Slang-Recognition-with-MECT-Model/main.py", line 701, in <module>
    test_label_list = predictor.predict(sentence)  # 预测结果
  File "/root/miniconda3/envs/normalpython/lib/python3.9/site-packages/fastNLP/core/predictor.py", line 64, in predict
    prediction = predict_func(**refined_batch_x)
  File "/root/autodl-tmp/Chinese-Slang-Recognition-with-MECT-Model/model.py", line 267, in forward
    bigrams_embed = self.bigram_embed(bigrams)
  File "/root/miniconda3/envs/normalpython/lib/python3.9/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/autodl-tmp/Chinese-Slang-Recognition-with-MECT-Model/Modules/StaticEmbedding.py", line 309, in forward
    words = self.words_to_words[words]
IndexError: index 94392 is out of bounds for dimension 0 with size 42889
 ```
 现在测得，对于贴吧数据集，使用微博的MECT模型，句子数量在5k是可以满足词表大小的，此时词表大小只有12396左右，微博MECT模型词表大小有42889