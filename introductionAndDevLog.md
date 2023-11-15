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

- 构建贴吧数据集

这里需要注意的一点是，贴吧数据集不需要做标记，也就是所有的字的标记都是O。这样的话直接用这个数据集套上已有的模型，就可以直接输出词向量，可以认为这个词向量是基于已有的MECT4CNER模型的拟合的模型构建的。

但是仅仅替换数据集好像不太行，因为fastnlp的Trainer训练的时候存的是模型的statedict，因此加载模型的时候需要把原模型初始化出来后再load
state dict，否则的话模型size不匹配，应该得想个别的办法弄弄

尝试用微博训练出来的模型去跑贴吧的数据集，然后发现报这个错：

 ```py
 TypeError: forward()
missing
4
required
positional
arguments: 'bigrams', 'seq_len', 'lex_num', and 'target'
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
| is_input    | False | False  | False   | False   | False    | False     | False   | False | False | True    | True  | True  |
| is_target   | False | False  | False   | False   | False    | False     | False   | False | False | False   | False | False |
| ignore_type |       |        |         |         |          |           |         |       |       | False   | False | False |
| pad_value   |       |        |         |         |          |           |         |       |       | 0       | 0     | 0     |

然后发现，上面报错的那四个变量，确实没有放到input里面去。。。。。。一下午时间就干了个这个。。。。。。。

同时，如果外接数据集过大的话，确实会出现词表溢出的情况，比如下面：

 ```py
 Traceback(most
recent
call
last):
File
"/root/autodl-tmp/Chinese-Slang-Recognition-with-MECT-Model/main.py", line
701, in < module >
test_label_list = predictor.predict(sentence)  # 预测结果
File
"/root/miniconda3/envs/normalpython/lib/python3.9/site-packages/fastNLP/core/predictor.py", line
64, in predict
prediction = predict_func(**refined_batch_x)
File
"/root/autodl-tmp/Chinese-Slang-Recognition-with-MECT-Model/model.py", line
267, in forward
bigrams_embed = self.bigram_embed(bigrams)
File
"/root/miniconda3/envs/normalpython/lib/python3.9/site-packages/torch/nn/modules/module.py", line
727, in _call_impl
result = self.forward(*input, **kwargs)
File
"/root/autodl-tmp/Chinese-Slang-Recognition-with-MECT-Model/Modules/StaticEmbedding.py", line
309, in forward
words = self.words_to_words[words]
IndexError: index
94392 is out
of
bounds
for dimension 0 with size 42889
 ```

现在测得，对于贴吧数据集，使用微博的MECT模型，句子数量在5k是可以满足词表大小的，此时词表大小只有12396左右，微博MECT模型词表大小有42889

- 构建聚类模块cluster.py

初步跑了一下聚类，发现在tieba数据集中和weibo数据集中，很多词语的向量在前两维上都有共性，比如下图为“你”这一字符在前两维的投影

![NItieba.png](md_cache/NItieba.png)

![NIweibo.png](md_cache/NIweibo.png)

# 11.8 日工作总结

对项目进行一些修补，同时训练了新的模型：

- 给服务器安装了字体，让聚类的结果能够以完整的中文图片展示

- 修改了字向量转化为词向量的处理逻辑，将词汇表中所有可行的词语都进行转换

这个工作的思路是，对于已有的句子，不能转化的就跳过，能转化的就转换。最后算出来的结果，发现只有7%的句子可以被转换，落实到数据上是达到了214443句，比之前的句子稍微多一些

- 训练了msra的mect模型

训练出这个模型的目的是看能不能扩大词表从而提升一下贴吧数据集中识别出来的词语的数量

为了达到这个目的，我还特地重新更新了一下数据集，保证除了onenote的数据集以外的其他数据集都是完整的

# 11.9 日工作总结

今日主要工作是需要构建聚类模块

- 构建聚类模块和评测指标

对于一个词语在多个数据集中的聚类结果，我们考虑如下问题

1：如何判断聚类结果的相似性

2：如果不相似，如何寻找其差异之处

其实第二个问题相对好解决，只要A数据集和B数据集的有差异，那么就说明A和B中的该词语的词义不同，然后就直接找聚类结果就行了

对于第一个问题，现在先提出两种方法：

第一：最简单的方法，就是判断最大可聚类数X的差值，X的值可以是直接取hardmax或者取softmax也可以

第二：对某个范围内的聚类值进行取样，这样对于一个词语在一个数据集中就会有一个聚类值序列。对于一个词语在两个数据集中的聚类值序列A和B，计算A和B的相似度即可。

对于第二种途径，先可以尝试采用[DWT时序匹配](https://liudongdong.blog.csdn.net/article/details/80344976)算法进行计算

[DWT时序匹配朴素代码在这里](https://zhuanlan.zhihu.com/p/87437065)

函数现在是构建好了，但是运行速度太慢了，一下午只能比较一丢丢东西，结果大概是这种情况：

```
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 25.60it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [04:34<00:00, 27.42s/it]
word 我 in dataset weibo and tieba with function 取样函数: difference is 252.0
word 我 in dataset weibo and tieba with function 最大化聚类函数: difference is 422
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 35.19it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [17:07<00:00, 102.79s/it]
word 了 in dataset weibo and tieba with function 取样函数: difference is 241.0
word 了 in dataset weibo and tieba with function 最大化聚类函数: difference is 291
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 35.58it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [03:47<00:00, 22.77s/it]
word 你 in dataset weibo and tieba with function 取样函数: difference is 258.0
word 你 in dataset weibo and tieba with function 最大化聚类函数: difference is 414
```

- 新建了基于PKU语料库的外来数据集

PKU语料库汇总起来确实大，能够达到1个多G的pkl，和tieba语料库相当了

测试结果也表示，PKU语料库中和tieba的重合字词集合中终于出现了词语，之前weibo和tieba的集合中都只有单个字的

- 使用msra的mect模型进行了数据集构建

实验证明，msra的模型词表就是大，同样是对于贴吧数据集，msra模型的词表能达到47%的成功率，微博模型的词表只有7%

# 11.10 日工作总结

- 提出第三种评测指标

第一种指标的缺点很明显，就不说了。

第二种指标也有缺点。首先，一个词语即使是暗语，在同一个数据集中它也可能会有正常的用法。因此反应在聚类上，暗语词汇的特征则是，同一个词语在不同的数据集上的聚类数量有微小且持久的差距（这里说的微小且持久，意思是，由于暗语词汇存在一些语义明显不同的上下文语义，因此暗语数据集的聚类结果总是比平常数据集的结果多一些，无论聚类的eps取多少）。但是如果使用第二种指标，那么这种微小且持久的差距是完全反应不出来的。因为第二种指标是使用的聚类数量为基准，对于每一个eps，第二种指标下，词语在数据集A和数据集B中的聚类差距应该是恒定的，因此第二种指标会认为这不是差距，从而忽略暗语特征。

因此现在需要结合上述问题提出第三种评测指标

- 提出对整套算法的评测方案

首先找数据集，数据集中人为的挑出一些暗语，然后以这些暗语为样本进行测试，计算F1值。

- 开发记忆化模块，使用缓存进行提速

详见Utils/AutoCache.py

关于提速这件事情，之前做过的调研表示，其实sklearn也是可以gpu加速的，使用cudf包就可以，但是问题在于，这个包一般的pip安装不了，得下源代码编译。

于是找了另外一个库：[sklearnex库](https://cloud.tencent.com/developer/article/2042042)

这个库据说也可以加速一些，但是是基于CPU加速的，服务器的GPU还是闲着的......

- 添加数据集

添加了暗语数据集，使用msra的模型能够达到100%的转化率，可以的

下一步是需要去找一些有明显俚语的数据集

# 11.11 日工作总结

- 构建了暗语数据集，查看了一下暗语数据集中出现次数大于1000的词语的大致情况如下

```text
，, 。, 的, 数据, 附件, ., 商品, 描述, .., 有, 交易, 发货, 不, 条, 是, 自动, 1, 拍, 编号, 了, 我, 你, 可以, 美金, 后, 万条, 
信息, 2, 在, 都, 请, 万, 姓名, 可, 和, 邮箱, 5, 退款, 3, ,, 购买, 需要, 电话, 网站, 就, 身份证, 年, 美国, 地址, 等, 复制, 
女性, 也, 价格, 下载, 截图, 出售, 为, 链接, 视频, 4, 最新, 月, 用户, 下, 虚拟, 看, 号, 自己, 资源, 格式, 拍下, 接受, 全国, 
出, 10, 个, 内, 一手, 测试, 性, 留言, 教程, 会, 时间, 到, 再, 客户, 一个, 包含, 一, 给, 含, 注册, 四, 要素, 份, 联系, 8, 
手机号, 资料, 部分, 使用, 被, 能, 平台, 直接, 放款, 6, 100, 下单, 付款, 具有, 提供, 一经, 微信, 人, 大, 做, 印度, 图片, 如果, 
问题, 一份, 手机, 网盘, 多, 没有, 用, 7, 来, 软件, 不要, 棋牌, 其他, 内容, 去, 详细, 站, 本人, 支持, 账号, 已经, 这个, 打包, 
```

# 11.13 日工作总结

构建了统计词语在数据集中出现过的句子的功能

同时选取了一些词语进行聚类

# 11.15日工作总结

跑完了选取的词语，结果存在clusterLog.txt中