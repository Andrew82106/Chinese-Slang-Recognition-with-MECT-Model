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