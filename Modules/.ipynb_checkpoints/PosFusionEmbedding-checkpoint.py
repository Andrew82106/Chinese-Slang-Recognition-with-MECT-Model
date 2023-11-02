import torch
from torch import nn
import torch.nn.functional as F


class PosFusionEmbedding(nn.Module):
    # 这是干啥用的哇
    # 这个模块在FLAT里面也有：https://github.com/LeeSureman/Flat-Lattice-Transformer/blob/master/V0/modules.py
    def __init__(self, pe, pe_ss, pe_ee, max_seq_len, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.pe_ss = pe_ss
        self.pe_ee = pe_ee
        self.pe = pe

        self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),
                                                nn.ReLU(inplace=True))

    def forward(self, pos_s, pos_e):
        batch = pos_s.size(0)

        pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2)
        # 这行代码创建了一个新的张量pos_ss，它是通过对pos_s进行操作得到的。
        # 具体地说，它计算了pos_s中每两个元素之间的差值，并将结果保存在pos_ss中。
        # unsqueeze函数用于在指定的维度上添加一个新的维度，-1表示在最后一个维度上添加。
        # 这个操作可能会用于计算位置之间的差异或距离。
        pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2)

        max_seq_len = pos_s.size(1)
        # 这行代码获取输入张量pos_s的第一个维度的大小，代表span序列的最大长度。
        pe_ss = self.pe_ss[(pos_ss).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        """
            这一句代码是对位置编码的生成和变形操作，让我详细解释：
    
            1. `pos_ss` 是一个3维张量，它包含了输入张量 `pos_s` 中每两个位置之间的差值。
            这个张量的形状通常是 `(batch, max_seq_len, max_seq_len)`，其中 `batch` 表示批次大小，`max_seq_len` 表示序列的最大长度。
    
            2. `(pos_ss).view(-1)` 这一部分是将 `pos_ss` 张量展平为一维张量。
            展平的操作会将原本的多维张量拉伸成一个一维向量，这有助于后续的索引操作。
    
            3. `self.max_seq_len` 是一个常数或者参数，它表示序列的最大长度。在这里，它被用来执行一个平移操作。
    
            4. `(pos_ss).view(-1) + self.max_seq_len` 这一部分将展平后的 `pos_ss` 张量中的每个元素都加上了 `self.max_seq_len`。
            这个操作的目的是对位置差值进行平移，将它们的范围映射到一个更大的范围内。
    
            5. `[self.pe_ss[(pos_ss).view(-1) + self.max_seq_len]` 这一部分是利用平移后的位置差值作为索引，从 `self.pe_ss` 中获取对应的位置编码。
            `self.pe_ss` 可能是一个预先计算好的位置编码张量，其中包含了各种位置信息的编码。
    
            6. `.view(size=[batch, max_seq_len, max_seq_len, -1])` 这一部分是将获取的位置编码张量重新变形为一个四维张量。
            具体地说，它将一维张量重新变形为形状为 `(batch, max_seq_len, max_seq_len, -1)` 的四维张量。
            这个操作的目的是将位置编码张量的形状与输入数据的形状相匹配，以便后续的操作。
    
            总结起来，这一句代码的主要作用是根据输入位置差值 `pos_ss`，从预先计算好的位置编码张量 `self.pe_ss` 中获取对应的位置编码，
            并将获取的位置编码张量变形为与输入数据相匹配的形状。
            这个操作用于为输入数据中的不同位置提供相应的位置编码信息，以帮助神经网络处理位置相关的特征。位置编码通常用于自注意力机制等模型中，
            以便模型能够理解序列中不同位置的关联和重要性。
        """
        pe_ee = self.pe_ee[(pos_ee).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])

        pe_2 = torch.cat([pe_ss, pe_ee], dim=-1)
        rel_pos_embedding = self.pos_fusion_forward(pe_2)

        return rel_pos_embedding
