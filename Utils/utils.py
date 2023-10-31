import collections
import math
import time
from copy import deepcopy

import datetime
import fitlog
import pytz
import torch
from fastNLP import FitlogCallback, Callback, EarlyStopError, logger


class EarlyStopCallback(Callback):
    r"""
    多少个epoch没有变好就停止训练，相关类 :class:`~fastNLP.core.callback.EarlyStopError`
    """

    def __init__(self, patience):
        r"""

        :param int patience: epoch的数量
        """
        super(EarlyStopCallback, self).__init__()
        self.patience = patience
        self.wait = 0

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if not is_better_eval:
            # current result is getting worse
            if self.wait == self.patience:
                raise EarlyStopError("Early stopping raised.")
            else:
                self.wait += 1
        else:
            self.wait = 0

    def on_exception(self, exception):
        if isinstance(exception, EarlyStopError):
            logger.info("Early Stopping triggered in epoch {}!".format(self.epoch))
        else:
            raise exception  # 抛出陌生Error


class MyFitlogCallback(FitlogCallback):
    def __init__(self, data=None, tester=None, log_loss_every=0, verbose=0, log_exception=False):
        super().__init__(data, tester, log_loss_every, verbose, log_exception)
        self.better_test_f = 0
        self.better_test_result = None

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if better_result:
            eval_result = deepcopy(eval_result)
            eval_result['step'] = self.step
            eval_result['epoch'] = self.epoch
            fitlog.add_best_metric(eval_result)
            # 保存最好的模型
            # torch.save(self.model.state_dict(), '/data/ws/radical_vec/model_parameter.pkl')
        fitlog.add_metric(eval_result, step=self.step, epoch=self.epoch)
        # if eval_result['SpanFPreRecMetric']['f'] < 0.2:
        #     raise EarlyStopError("Early stopping raised.")
        if len(self.testers) > 0:
            for key, tester in self.testers.items():
                try:
                    eval_result = tester.test()
                    if self.verbose != 0:
                        self.pbar.write("FitlogCallback evaluation on {}:".format(key))
                        self.pbar.write(tester._format_eval_results(eval_result))
                    fitlog.add_metric(eval_result, name=key, step=self.step, epoch=self.epoch)
                    if better_result:
                        self.better_test_f = eval_result['SpanFPreRecMetric']['f']
                        fitlog.add_best_metric(eval_result, name=key)
                except Exception as e:
                    self.pbar.write("Exception happens when evaluate on DataSet named `{}`.".format(key))
                    raise e

    def on_train_end(self):
        fitlog.finish()

    def on_exception(self, exception):
        fitlog.finish(status=1)
        if self._log_exception:
            fitlog.add_other(repr(exception), name='except_info')


def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    rel pos init:
    如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
    如果是1，那么就按-max_len,max_len来初始化
    """
    num_embeddings = 2*max_seq_len+1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len,max_seq_len+1, dtype=torch.float).unsqueeze(1)*emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb

"""
这段代码用于生成位置嵌入（Positional Embeddings），通常用于自注意力模型（比如Transformer）中，以帮助模型捕捉序列中不同位置的信息。位置嵌入在模型中与词嵌入（Word Embeddings）一起使用，以使模型能够理解输入序列中的词语或标记在序列中的位置。

具体来说，这段代码的功能如下：

1. `num_embeddings = 2*max_seq_len+1`：计算位置嵌入的总数量。这里的 `max_seq_len` 表示序列的最大长度，而 `num_embeddings` 表示要生成的位置嵌入的数量。通常，位置嵌入会包括负数、零以及正数位置的编码，因此总数量是原始序列长度的两倍再加一。

2. `half_dim = embedding_dim // 2`：计算位置嵌入的维度的一半。这是因为通常位置嵌入是一个包含正弦和余弦函数值的矩阵，而这两个函数分别被赋予了一半的维度。

3. 计算正弦和余弦函数的值，并按照指定的初始化方式（`rel_pos_init`）进行初始化。具体来说：
   - 如果 `rel_pos_init` 为0，那么从 `-max_len` 到 `max_len` 的相对位置编码矩阵就按 `0` 到 `2 * max_len` 来初始化。
   - 如果 `rel_pos_init` 为1，那么从 `-max_len` 到 `max_len` 的相对位置编码矩阵就按 `-max_len` 到 `max_len` 来初始化。

4. 将正弦和余弦函数的值连接在一起，得到位置嵌入矩阵。这个矩阵的形状是 `(num_embeddings, embedding_dim)`，其中每一行代表一个不同位置的嵌入向量。

5. 如果 `embedding_dim` 是奇数，那么在嵌入矩阵的末尾添加一个零列，以确保嵌入向量的维度与指定的 `embedding_dim` 一致。

6. 如果 `padding_idx` 参数不为 `None`，则将指定位置的嵌入向量置为零。这通常用于处理序列中的填充标记。

7. 最后，返回生成的位置嵌入矩阵。

总之，这段代码用于生成用于表示序列位置信息的位置嵌入矩阵，这些位置嵌入可以与词嵌入一起输入到深度学习模型中，以帮助模型理解输入序列中不同位置的关系和重要性。
"""


def get_crf_zero_init(label_size, include_start_end_trans=False, allowed_transitions=None,
                 initial_method=None):
    import torch.nn as nn
    from fastNLP.modules import ConditionalRandomField
    crf = ConditionalRandomField(label_size, include_start_end_trans)

    crf.trans_m = nn.Parameter(torch.zeros(size=[label_size, label_size], requires_grad=True))
    if crf.include_start_end_trans:
        crf.start_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
        crf.end_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
    return crf


def norm_static_embedding(x,norm=1):
    with torch.no_grad():
        x.embedding.weight /= (torch.norm(x.embedding.weight, dim=1, keepdim=True) + 1e-12)
        x.embedding.weight *= norm


def get_bigrams(words):
    result = []
    for i, w in enumerate(words):
        if i != len(words)-1:
            result.append(words[i]+words[i+1])
        else:
            result.append(words[i]+'<end>')

    return result


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_w = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self,w):

        current = self.root
        for c in w:
            current = current.children[c]

        current.is_w = True

    def search(self,w):
        '''

        :param w:
        :return:
        -1:not w route
        0:subroute but not word
        1:subroute and word
        '''
        current = self.root

        for c in w:
            current = current.children.get(c)

            if current is None:
                return -1

        if current.is_w:
            return 1
        else:
            return 0

    def get_lexicon(self,sentence):
        result = []
        for i in range(len(sentence)):
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                if current is None:
                    break

                if current.is_w:
                    result.append([i,j,sentence[i:j+1]])

        return result


def get_skip_path(chars,w_trie):
    sentence = ''.join(chars)
    result = w_trie.get_lexicon(sentence)

    return result


def print_info(*inp, islog=True, sep=' '):
    from fastNLP import logger
    if islog:
        print(*inp,sep=sep)
    else:
        inp = sep.join(map(str,inp))
        logger.info(inp)


def get_peking_time():

    tz = pytz.timezone('Asia/Shanghai')  # 东八区

    t = datetime.datetime.fromtimestamp(int(time.time()), tz).strftime('%Y_%m_%d_%H_%M_%S')
    return t
