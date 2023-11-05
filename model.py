import torch
from fastNLP import seq_len_to_mask
from torch import nn

from Modules.MyDropout import MyDropout
from Modules.TransformerEncoder import TransformerEncoder
from Utils.utils import get_crf_zero_init, get_embedding


class MECTNER(nn.Module):
    def __init__(self, lattice_embed, bigram_embed, components_embed, hidden_size,
                 k_proj, q_proj, v_proj, r_proj,
                 label_size, max_seq_len, dropout, dataset, ff_size):
        super().__init__()

        self.dataset = dataset

        self.lattice_embed = lattice_embed
        self.bigram_embed = bigram_embed
        self.components_embed = components_embed
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        print(f"lattice_embed:{lattice_embed}")
        print(f"bigram_embed:{bigram_embed}")
        print(f"components_embed:{components_embed}")
        # exit(0)
        # 超参数
        self.rel_pos_init = 0
        self.learnable_position = False
        self.num_heads = 8
        self.layer_preprocess_sequence = ""
        self.layer_postprocess_sequence = "an"
        self.dropout = dropout
        self.scaled = False
        self.ff_size = ff_size
        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = True
        self.ff_activate = 'relu'
        self.use_pytorch_dropout = 0
        self.embed_dropout = MyDropout(self.dropout['embed'])
        self.gaz_dropout = MyDropout(self.dropout['gaz'])
        self.output_dropout = MyDropout(self.dropout['output'])

        pe = get_embedding(max_seq_len, self.hidden_size, rel_pos_init=self.rel_pos_init)
        self.pe = nn.Parameter(pe, requires_grad=self.learnable_position)
        self.pe_ss = self.pe
        self.pe_ee = self.pe

        self.lex_input_size = self.lattice_embed.embed_size
        self.bigram_size = self.bigram_embed.embed_size
        self.components_embed_size = self.components_embed.embed_size
        self.char_input_size = self.lex_input_size + self.bigram_size

        self.char_proj = nn.Linear(self.char_input_size, self.hidden_size)
        self.lex_proj = nn.Linear(self.lex_input_size, self.hidden_size)
        self.components_proj = nn.Linear(self.components_embed_size, self.hidden_size)

        self.char_encoder = TransformerEncoder(self.hidden_size, self.num_heads,
                                               dataset=self.dataset,
                                               layer_preprocess_sequence=self.layer_preprocess_sequence,
                                               layer_postprocess_sequence=self.layer_postprocess_sequence,
                                               dropout=self.dropout,
                                               scaled=self.scaled,
                                               ff_size=self.ff_size,
                                               max_seq_len=self.max_seq_len,
                                               pe=self.pe,
                                               pe_ss=self.pe_ss,
                                               pe_ee=self.pe_ee,
                                               ff_activate=self.ff_activate,
                                               use_pytorch_dropout=self.use_pytorch_dropout)

        self.radical_encoder = TransformerEncoder(self.hidden_size, self.num_heads,
                                                  dataset=self.dataset,
                                                  layer_preprocess_sequence=self.layer_preprocess_sequence,
                                                  layer_postprocess_sequence=self.layer_postprocess_sequence,
                                                  dropout=self.dropout,
                                                  scaled=self.scaled,
                                                  ff_size=self.ff_size,
                                                  max_seq_len=self.max_seq_len,
                                                  pe=self.pe,
                                                  pe_ss=self.pe_ss,
                                                  pe_ee=self.pe_ee,
                                                  ff_activate=self.ff_activate,
                                                  use_pytorch_dropout=self.use_pytorch_dropout)

        self.output = nn.Linear(self.hidden_size * 2, self.label_size)

        self.crf = get_crf_zero_init(self.label_size)

    def forward(self, lattice, bigrams, seq_len, lex_num, pos_s, pos_e, target):
        batch_size = lattice.size(0)
        max_seq_len_and_lex_num = lattice.size(1)
        max_seq_len = bigrams.size(1)

        raw_embed = self.lattice_embed(lattice)

        char_mask = seq_len_to_mask(seq_len, max_len=max_seq_len_and_lex_num).bool()
        char = lattice.masked_fill_(~char_mask, 0)
        components_embed = self.components_embed(char)
        components_embed.masked_fill_(~(char_mask).unsqueeze(-1), 0)
        components_embed = self.components_proj(components_embed)
        bigrams_embed = self.bigram_embed(bigrams)
        bigrams_embed = torch.cat([bigrams_embed,
                                   torch.zeros(size=[batch_size, max_seq_len_and_lex_num - max_seq_len,
                                                     self.bigram_size]).to(bigrams_embed)], dim=1)
        raw_embed_char = torch.cat([raw_embed, bigrams_embed], dim=-1)

        raw_embed_char = self.embed_dropout(raw_embed_char)
        raw_embed = self.gaz_dropout(raw_embed)

        embed_char = self.char_proj(raw_embed_char)
        char_mask = seq_len_to_mask(seq_len, max_len=max_seq_len_and_lex_num).bool()
        embed_char.masked_fill_(~(char_mask.unsqueeze(-1)), 0)

        embed_lex = self.lex_proj(raw_embed)
        lex_mask = (seq_len_to_mask(seq_len + lex_num).bool() ^ char_mask.bool())
        embed_lex.masked_fill_(~lex_mask.unsqueeze(-1), 0)

        assert char_mask.size(1) == lex_mask.size(1)
        embedding = embed_char + embed_lex

        char_encoded = self.char_encoder(components_embed, embedding, embedding, seq_len, lex_num=lex_num, pos_s=pos_s,
                                         pos_e=pos_e)
        radical_encoded = self.radical_encoder(embedding, components_embed, components_embed, seq_len,
                                               lex_num=lex_num, pos_s=pos_s, pos_e=pos_e)

        # char_encoded和radical_encoded就是论文中CrossTransformer module的Transformer encoder实现

        fusion = torch.cat([radical_encoded, char_encoded], dim=-1)
        output = self.output_dropout(fusion)
        output = output[:, :max_seq_len, :]
        pred = self.output(output)

        mask = seq_len_to_mask(seq_len).bool()

        if self.training:
            loss = self.crf(pred, target, mask).mean(dim=0)
            return {'loss': loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, mask)
            result = {'pred': pred}

            return result

        
class CSR_MECTNER(nn.Module):
    def __init__(self, lattice_embed, bigram_embed, components_embed, hidden_size,
                 k_proj, q_proj, v_proj, r_proj,
                 label_size, max_seq_len, dropout, dataset, ff_size):
        """
        1：
            结合main.py里面的代码
            lattice_embed, bigram_embed, components_embed,这三个变量应该是在CNNRadicalLevelEmbedding函数中构建的三个模块
            其分别属于：
                <class 'Modules.StaticEmbedding.StaticEmbedding'>
                <class 'Modules.StaticEmbedding.StaticEmbedding'>
                <class 'Modules.CNNRadicalLevelEmbedding.CNNRadicalLevelEmbedding'>
            这三个类
            具体的功能应该是字embedding，双字embedding和部首embedding模块
        """
        super().__init__()

        self.dataset = dataset

        self.lattice_embed = lattice_embed
        self.bigram_embed = bigram_embed
        self.components_embed = components_embed
        
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        print(f"lattice_embed:{lattice_embed}")
        print(f"bigram_embed:{bigram_embed}")
        print(f"components_embed:{components_embed}")
        # exit(0)
        # 超参数
        self.rel_pos_init = 0
        self.learnable_position = False
        self.num_heads = 8
        self.layer_preprocess_sequence = ""
        self.layer_postprocess_sequence = "an"
        self.dropout = dropout
        self.scaled = False
        self.ff_size = ff_size
        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = True
        self.ff_activate = 'relu'
        self.use_pytorch_dropout = 0
        self.embed_dropout = MyDropout(self.dropout['embed'])
        self.gaz_dropout = MyDropout(self.dropout['gaz'])
        self.output_dropout = MyDropout(self.dropout['output'])

        pe = get_embedding(max_seq_len, self.hidden_size, rel_pos_init=self.rel_pos_init)
        self.pe = nn.Parameter(pe, requires_grad=self.learnable_position)
        self.pe_ss = self.pe
        self.pe_ee = self.pe

        self.lex_input_size = self.lattice_embed.embed_size
        self.bigram_size = self.bigram_embed.embed_size
        self.components_embed_size = self.components_embed.embed_size
        self.char_input_size = self.lex_input_size + self.bigram_size

        self.char_proj = nn.Linear(self.char_input_size, self.hidden_size)
        self.lex_proj = nn.Linear(self.lex_input_size, self.hidden_size)
        self.components_proj = nn.Linear(self.components_embed_size, self.hidden_size)

        self.char_encoder = TransformerEncoder(self.hidden_size, self.num_heads,
                                               dataset=self.dataset,
                                               layer_preprocess_sequence=self.layer_preprocess_sequence,
                                               layer_postprocess_sequence=self.layer_postprocess_sequence,
                                               dropout=self.dropout,
                                               scaled=self.scaled,
                                               ff_size=self.ff_size,
                                               max_seq_len=self.max_seq_len,
                                               pe=self.pe,
                                               pe_ss=self.pe_ss,
                                               pe_ee=self.pe_ee,
                                               ff_activate=self.ff_activate,
                                               use_pytorch_dropout=self.use_pytorch_dropout)

        self.radical_encoder = TransformerEncoder(self.hidden_size, self.num_heads,
                                                  dataset=self.dataset,
                                                  layer_preprocess_sequence=self.layer_preprocess_sequence,
                                                  layer_postprocess_sequence=self.layer_postprocess_sequence,
                                                  dropout=self.dropout,
                                                  scaled=self.scaled,
                                                  ff_size=self.ff_size,
                                                  max_seq_len=self.max_seq_len,
                                                  pe=self.pe,
                                                  pe_ss=self.pe_ss,
                                                  pe_ee=self.pe_ee,
                                                  ff_activate=self.ff_activate,
                                                  use_pytorch_dropout=self.use_pytorch_dropout)

        self.output = nn.Linear(self.hidden_size * 2, self.label_size)

        self.crf = get_crf_zero_init(self.label_size)

    def forward(self, lattice, bigrams, seq_len, lex_num, pos_s, pos_e, target):
        """
        fastnlp.core.Trainer的forward的参数来自于datasets变量中被设为input的field，因此这里的lattice、bigrams等变量的含义和main.py中的datasets变量中的函数是一样的。

        """
        batch_size = lattice.size(0)
        max_seq_len_and_lex_num = lattice.size(1)
        max_seq_len = bigrams.size(1)

        raw_embed = self.lattice_embed(lattice)
        # 对lattice数据进行embedding操作

        char_mask = seq_len_to_mask(seq_len, max_len=max_seq_len_and_lex_num).bool()
        char = lattice.masked_fill_(~char_mask, 0)
        # 使用 seq_len_to_mask 函数生成掩码 char_mask，并根据生成的掩码来更新 lattice 张量。
        
        components_embed = self.components_embed(char)
        components_embed.masked_fill_(~(char_mask).unsqueeze(-1), 0)
        components_embed = self.components_proj(components_embed)
        # self.components_proj = nn.Linear(self.components_embed_size, self.hidden_size)
        # 用一个线性全连接层把部首的size连接到hidden_size中
        # 构建部首embedding结果
        
        bigrams_embed = self.bigram_embed(bigrams)
        bigrams_embed = torch.cat([bigrams_embed,
                                   torch.zeros(size=[batch_size, max_seq_len_and_lex_num - max_seq_len,
                                                     self.bigram_size]).to(bigrams_embed)], dim=1)
        # 构建bigrams embedding结果
        
        raw_embed_char = torch.cat([raw_embed, bigrams_embed], dim=-1)
        # 将lattice数据的结果和bigrams的结果融合起来
        # 可是lattice数据的结果里面已经有bigrams的结果了，为什么还融合一遍

        raw_embed_char = self.embed_dropout(raw_embed_char)
        raw_embed = self.gaz_dropout(raw_embed)
        # MyDropout(self.dropout['embed'])
        # MyDropout(self.dropout['gaz'])
        # 这里就是对raw_embed以embed和gaz的概率进行dropout

        embed_char = self.char_proj(raw_embed_char)
        # self.char_proj = nn.Linear(self.char_input_size, self.hidden_size)
        # 和components_proj类似，用一个线性全连接层把raw_embed_char的size连接到hidden_size中
        char_mask = seq_len_to_mask(seq_len, max_len=max_seq_len_and_lex_num).bool()
        embed_char.masked_fill_(~(char_mask.unsqueeze(-1)), 0)
        # 对embed_char进行mask

        embed_lex = self.lex_proj(raw_embed)
        # self.lex_proj = nn.Linear(self.lex_input_size, self.hidden_size)
        # 用一个线性全连接层把raw_embed的size连接到hidden_size中
        lex_mask = (seq_len_to_mask(seq_len + lex_num).bool() ^ char_mask.bool())
        embed_lex.masked_fill_(~lex_mask.unsqueeze(-1), 0)
        # 对embed_lex 进行mask

        assert char_mask.size(1) == lex_mask.size(1)
        embedding = embed_char + embed_lex

        char_encoded = self.char_encoder(components_embed, embedding, embedding, seq_len, lex_num=lex_num, pos_s=pos_s,
                                         pos_e=pos_e)
        radical_encoded = self.radical_encoder(embedding, components_embed, components_embed, seq_len,
                                               lex_num=lex_num, pos_s=pos_s, pos_e=pos_e)

        # char_encoded和radical_encoded就是论文中CrossTransformer module的Transformer encoder实现

        fusion = torch.cat([radical_encoded, char_encoded], dim=-1)
        output = self.output_dropout(fusion)
        # self.output_dropout = MyDropout(self.dropout['output'])
        output = output[:, :max_seq_len, :]
        """
        您给出的代码片段 output = output[:, :max_seq_len, :] 是Python中使用切片(slice)操作一个多维数组（例如：NumPy数组）的常见方式。此代码的目的是截取数组的一部分。
        output[:, :max_seq_len, :]: 这里的切片操作针对的是一个三维数组。
        :: 表示选取该维度的所有元素。
        :max_seq_len: 表示在该维度上切片，从开始到max_seq_len位置。
        最后一个:也表示选取该维度的所有元素。
        假设 output 是一个形状为 (batch_size, seq_len, features) 的三维数组，那么该切片操作将数组截断到 max_seq_len 长度，以 max_seq_len 为截断长度，使得在序列长度这一维度上不超过 max_seq_len。
        """
        seq_output = output
        # 我感觉这个应该是最理想的输出结果，这个里面应该包含了每一个字的字向量
        pred = self.output(output)
        # self.output = nn.Linear(self.hidden_size * 2, self.label_size)
        # 也就是说，self.output的长度就是self.hidden_size * 2
        
        mask = seq_len_to_mask(seq_len).bool()

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