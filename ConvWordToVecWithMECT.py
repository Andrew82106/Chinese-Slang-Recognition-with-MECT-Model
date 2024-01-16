import os

import fitlog
import tqdm
from Modules.CNNRadicalLevelEmbedding import CNNRadicalLevelEmbedding
from Utils.load_data import *
from Utils.paths import *
from Utils.utils import norm_static_embedding, MyFitlogCallback, print_info, get_peking_time
from model import MECTNER, CSR_MECTNER
from Modules.CharacterToWord import CTW
from Modules.WordCut import ChineseTokenizer
from Utils.ToAndFromPickle import *
from Utils.AutoCache import Cache

cache = Cache()
use_fitlog = False
if not use_fitlog:
    fitlog.debug()
fitlog.set_log_dir('logs')
load_dataset_seed = 100
fitlog.add_hyper(load_dataset_seed, 'load_dataset_seed')
fitlog.set_rng_seed(load_dataset_seed)
import sys

sys.path.append('../')
import torch
import collections
import torch.optim as optim
import torch.nn as nn
from fastNLP.core.metrics import SpanFPreRecMetric, AccuracyMetric
from fastNLP.core.callback import WarmupCallback, GradientClipCallback, EarlyStopCallback
from fastNLP import LRScheduler, DataSetIter, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
from fastNLP import logger
import pprint


def preprocess(args, outdatasetPath=test_path, refresh=False):
    """
    该函数使用了MECT4CNER，将outdatasetPath中的文本变成向量组，作为返回结果
    这里需要注意的是，只要test.pkl存在，并且没有强制要求刷新，那就不要再跑一次这份代码了，很蠢
    """
    if os.path.exists(test_vector) and (refresh is False):
        print(f"file {test_vector} exists, no need to fresh")
        return load_from_pickle(test_vector)

    if args.ff_dropout_2 < 0:
        args.ff_dropout_2 = args.ff_dropout
    over_all_dropout = -1
    if over_all_dropout > 0:
        args.embed_dropout = over_all_dropout
        args.output_dropout = over_all_dropout
        args.pre_dropout = over_all_dropout
        args.post_dropout = over_all_dropout
        args.ff_dropout = over_all_dropout
        args.attn_dropout = over_all_dropout

    if args.lattice and args.use_rel_pos and args.update_every == 1:
        args.train_clip = True

    now_time = get_peking_time()
    logger.add_file('log/{}'.format(now_time), level='info')
    if args.test_batch == -1:
        args.test_batch = args.batch // 2
    fitlog.add_hyper(now_time, 'time')
    refresh_data = False
    for k, v in args.__dict__.items():
        print_info('{}:{}'.format(k, v))

    args.dataset = 'msra'
    args.extra_datasets = 'test'
    ###################################### para setting ######################################

    raw_dataset_cache_name = os.path.join('cache', args.dataset +
                                          '_trainClip{}'.format(args.train_clip)
                                          + 'bgminfreq_{}'.format(args.bigram_min_freq)
                                          + 'char_min_freq_{}'.format(args.char_min_freq)
                                          + 'word_min_freq_{}'.format(args.word_min_freq)
                                          + 'only_train_min_freq{}'.format(args.only_train_min_freq)
                                          + 'number_norm{}'.format(args.number_normalized)
                                          + 'load_dataset_seed{}'.format(load_dataset_seed)
                                          )

    if args.dataset == 'ontonotes':
        datasets, vocabs, embeddings = load_ontonotes4ner(ontonote4ner_cn_path, yangjie_rich_pretrain_unigram_path,
                                                          yangjie_rich_pretrain_bigram_path,
                                                          _refresh=refresh_data, index_token=False,
                                                          train_clip=args.train_clip,
                                                          _cache_fp=raw_dataset_cache_name,
                                                          char_min_freq=args.char_min_freq,
                                                          bigram_min_freq=args.bigram_min_freq,
                                                          only_train_min_freq=args.only_train_min_freq
                                                          )
    elif args.dataset == 'resume':
        datasets, vocabs, embeddings = load_resume_ner(resume_ner_path, yangjie_rich_pretrain_unigram_path,
                                                       yangjie_rich_pretrain_bigram_path,
                                                       _refresh=refresh_data, index_token=False,
                                                       _cache_fp=raw_dataset_cache_name,
                                                       char_min_freq=args.char_min_freq,
                                                       bigram_min_freq=args.bigram_min_freq,
                                                       only_train_min_freq=args.only_train_min_freq
                                                       )
    elif args.dataset == 'weibo':
        if args.label == 'ne':
            raw_dataset_cache_name = 'ne' + raw_dataset_cache_name
        elif args.label == 'nm':
            raw_dataset_cache_name = 'nm' + raw_dataset_cache_name
        datasets, vocabs, embeddings = load_weibo_ner(weibo_ner_path, yangjie_rich_pretrain_unigram_path,
                                                      yangjie_rich_pretrain_bigram_path,
                                                      _refresh=refresh_data, index_token=False,
                                                      _cache_fp=raw_dataset_cache_name,
                                                      char_min_freq=args.char_min_freq,
                                                      bigram_min_freq=args.bigram_min_freq,
                                                      only_train_min_freq=args.only_train_min_freq
                                                      )
    elif args.dataset == 'msra':
        args.train_clip = True
        datasets, vocabs, embeddings = load_msra_ner_1(msra_ner_cn_path, yangjie_rich_pretrain_unigram_path,
                                                       yangjie_rich_pretrain_bigram_path,
                                                       _refresh=refresh_data, index_token=False,
                                                       train_clip=args.train_clip,
                                                       _cache_fp=raw_dataset_cache_name,
                                                       char_min_freq=args.char_min_freq,
                                                       bigram_min_freq=args.bigram_min_freq,
                                                       only_train_min_freq=args.only_train_min_freq
                                                       )
    else:
        datasets, vocabs, embeddings = load_demo(demo_ner_path, yangjie_rich_pretrain_unigram_path,
                                                 yangjie_rich_pretrain_bigram_path,
                                                 _refresh=refresh_data, index_token=False, train_clip=args.train_clip,
                                                 _cache_fp=raw_dataset_cache_name,
                                                 char_min_freq=args.char_min_freq,
                                                 bigram_min_freq=args.bigram_min_freq,
                                                 only_train_min_freq=args.only_train_min_freq
                                                 )

    if args.gaz_dropout < 0:
        args.gaz_dropout = args.embed_dropout

    args.hidden = args.head_dim * args.head
    args.ff = args.hidden * args.ff
    args.q_proj = 1
    args.k_proj = 0
    args.v_proj = 1

    if args.dataset == 'resume':
        args.ff_dropout = 0.1
        args.ff_dropout_2 = 0.1
        args.radical_dropout = 0.1
        args.char_dropout = 0.3
        args.head_dim = 16
        args.ff = 384
        args.hidden = 128
        args.warmup = 0.05
        args.lr = 0.0014
        args.components_embed_lr_rate = 0.0018
        args.momentum = 0.9
        args.epoch = 50
        args.seed = 9249
    elif args.dataset == 'ontonotes':
        args.ff_dropout = 0.2
        args.ff_dropout_2 = 0.1
        args.radical_dropout = 0.4
        args.head_dim = 20
        args.ff = 480
        args.warmup = 0.1
        args.lr = 0.0005
        args.components_embed_lr_rate = 0.0005
        args.momentum = 0.9
        args.update_every = 1
        args.epoch = 100
    elif args.dataset == 'msra':
        args.ff_dropout = 0.1
        args.ff_dropout_2 = 0.1
        args.radical_dropout = 0.2
        args.char_dropout = 0.1
        args.head_dim = 20
        args.ff = 480
        args.warmup = 0.1
        args.lr = 0.0014
        args.components_embed_lr_rate = 0.0012
        args.momentum = 0.85
        args.update_every = 1
        args.epoch = 100
    elif args.dataset == 'demo':
        args.ff_dropout = 0.1
        args.ff_dropout_2 = 0.1
        args.radical_dropout = 0.2
        args.char_dropout = 0.1
        args.head_dim = 20
        args.ff = 480
        args.warmup = 0.1
        args.lr = 0.0014
        args.components_embed_lr_rate = 0.0012
        args.momentum = 0.85
        args.update_every = 1
        args.epoch = 100
    elif args.dataset == 'weibo':
        args.ff_dropout = 0.2
        args.ff_dropout_2 = 0.4
        args.gaz_dropout = 0.5
        args.head_dim = 16
        args.ff = 384
        args.hidden = 128
        args.radical_dropout = 0.2
        args.warmup = 0.3
        args.lr = 0.0018
        args.components_embed_lr_rate = 0.0014
        args.momentum = 0.9
        args.epoch = 50
    ###################################### datasets setting ######################################

    print('用的词表的路径:{}'.format(yangjie_rich_pretrain_word_path))

    w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                                  _refresh=refresh_data,
                                                  _cache_fp='cache/{}'.format(args.lexicon_name))
    # w_list就是一个词表
    cache_name = os.path.join('cache', (args.dataset + '_lattice' + '_only_train{}' +
                                        '_trainClip{}' + '_norm_num{}'
                                        + 'char_min_freq{}' + 'bigram_min_freq{}' + 'word_min_freq{}' + 'only_train_min_freq{}'
                                        + 'number_norm{}' + 'lexicon_{}' + 'load_dataset_seed{}')
                              .format(args.only_lexicon_in_train,
                                      args.train_clip, args.number_normalized, args.char_min_freq,
                                      args.bigram_min_freq, args.word_min_freq, args.only_train_min_freq,
                                      args.number_normalized, args.lexicon_name, load_dataset_seed))
    if args.dataset == 'weibo':
        if args.label == 'ne':
            cache_name = 'ne' + cache_name
        elif args.label == 'nm':
            cache_name = 'nm' + cache_name


    datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(datasets, vocabs, embeddings,
                                                                  w_list, yangjie_rich_pretrain_word_path,
                                                                  _refresh=refresh_data, _cache_fp=cache_name,
                                                                  only_lexicon_in_train=args.only_lexicon_in_train,
                                                                  word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                                  number_normalized=args.number_normalized,
                                                                  lattice_min_freq=args.lattice_min_freq,
                                                                  only_train_min_freq=args.only_train_min_freq)

    max_seq_len = max(*map(lambda x: max(x['seq_len']), datasets.values()))

    for k, v in datasets.items():
        if args.lattice:
            v.set_input('lattice', 'bigrams', 'seq_len', 'target')
            v.set_input('lex_num', 'pos_s', 'pos_e')
            v.set_target('target', 'seq_len')
        else:
            v.set_input('chars', 'bigrams', 'seq_len', 'target')
            v.set_target('target', 'seq_len')

    if args.norm_embed > 0:
        print('embedding:{}'.format(embeddings['char'].embedding.weight.size()))
        print('norm embedding')
        for k, v in embeddings.items():
            norm_static_embedding(v, args.norm_embed)

    if args.norm_lattice_embed > 0:
        print('embedding:{}'.format(embeddings['lattice'].embedding.weight.size()))
        print('norm lattice embedding')
        for k, v in embeddings.items():
            print(k, v)
            norm_static_embedding(v, args.norm_embed)

    dropout = collections.defaultdict(int)
    dropout['embed'] = args.embed_dropout
    dropout['gaz'] = args.gaz_dropout
    dropout['output'] = args.output_dropout
    dropout['pre'] = args.pre_dropout
    dropout['post'] = args.post_dropout
    dropout['ff'] = args.ff_dropout
    dropout['ff_2'] = args.ff_dropout_2
    dropout['attn'] = args.attn_dropout

    fitlog.set_rng_seed(args.seed)
    fitlog.add_hyper(args)

    """偏旁部首"""
    embeddings['components'] = CNNRadicalLevelEmbedding(vocab=vocabs['lattice'], embed_size=50, char_emb_size=30,
                                                        filter_nums=[30],
                                                        kernel_sizes=[3], char_dropout=args.char_dropout,
                                                        dropout=args.radical_dropout, pool_method='max'
                                                        , include_word_start_end=False, min_char_freq=1)


    print("finish embeddings model!")
    model_old = MECTNER(embeddings['lattice'], embeddings['bigram'], embeddings['components'], args.hidden,
                        k_proj=args.k_proj, q_proj=args.q_proj, v_proj=args.v_proj, r_proj=args.r_proj,
                        label_size=len(vocabs['label']), max_seq_len=max_seq_len,
                        dropout=dropout, dataset=args.dataset, ff_size=args.ff)

    model_pro = CSR_MECTNER(embeddings['lattice'], embeddings['bigram'], embeddings['components'], args.hidden,
                            k_proj=args.k_proj, q_proj=args.q_proj, v_proj=args.v_proj, r_proj=args.r_proj,
                            label_size=len(vocabs['label']), max_seq_len=max_seq_len,
                            dropout=dropout, dataset=args.dataset, ff_size=args.ff)

    model = model_pro

    print("test:", vocabs['lattice'].to_word(24))
    for n, p in model.named_parameters():
        print('{}:{}'.format(n, p.size()))

    with torch.no_grad():
        print_info('{}init pram{}'.format('*' * 15, '*' * 15))
        for n, p in model.named_parameters():
            if 'embedding' not in n and 'pos' not in n and 'pe' not in n \
                    and 'bias' not in n and 'crf' not in n and 'randomAttention' not in n and p.dim() > 1:
                try:
                    if args.init == 'uniform':
                        nn.init.xavier_uniform_(p)
                        print_info('xavier uniform init:{}'.format(n))
                    elif args.init == 'norm':
                        print_info('xavier norm init:{}'.format(n))
                        nn.init.xavier_normal_(p)
                except:
                    print_info(n)
                    exit(1208)
        print_info('{}init pram{}'.format('*' * 15, '*' * 15))

    acc_metric = AccuracyMetric(pred='pred', target='target', seq_len='seq_len', )
    acc_metric.set_metric_name('label_acc')

    bigram_embedding_param = list(model.bigram_embed.parameters())
    gaz_embedding_param = list(model.lattice_embed.parameters())
    components_embed_param = list(model.components_embed.parameters())
    embedding_param = bigram_embedding_param
    embedding_param = embedding_param + gaz_embedding_param
    embedding_param_ids = list(map(id, embedding_param + components_embed_param))
    non_embedding_param = list(filter(lambda x: id(x) not in embedding_param_ids, model.parameters()))

    param_ = [{'params': non_embedding_param},
              {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate},
              {'params': components_embed_param, 'lr': args.components_embed_lr_rate}]

    optimizer = optim.SGD(param_, lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if ('msra' in args.dataset) or ('demo' in args.dataset):
        datasets['dev'] = datasets['test']

    fitlog_evaluate_dataset = {'test': datasets['test']}
    evaluate_callback = MyFitlogCallback(fitlog_evaluate_dataset, verbose=1)
    lrschedule_callback = LRScheduler(lr_scheduler=LambdaLR(optimizer, lambda epoch: 1 / (1 + 0.05 * epoch)))
    clip_callback = GradientClipCallback(clip_type='value', clip_value=5)

    callbacks = [evaluate_callback, lrschedule_callback, clip_callback, WarmupCallback(warmup=args.warmup)]

    print("INFO:: Load Model")
    model_path = os.path.join(rootPth, 'model/best_CSR_MECTNER_f_msra')
    states = torch.load(model_path).state_dict()
    model.load_state_dict(states)
    from fastNLP.core.predictor import Predictor
    import random
    import time

    random.seed(time.time())

    predictor = Predictor(model)  # 这里的model是加载权重之后的model
    print(">>>>>>>成功加载模型<<<<<<<")

    raw_dataset_cache_name1 = os.path.join('cache', args.extra_datasets +
                                           '_trainClip{}'.format(args.train_clip)
                                           + 'bgminfreq_{}'.format(args.bigram_min_freq)
                                           + 'char_min_freq_{}'.format(args.char_min_freq)
                                           + 'word_min_freq_{}'.format(args.word_min_freq)
                                           + 'only_train_min_freq{}'.format(args.only_train_min_freq)
                                           + 'number_norm{}'.format(args.number_normalized)
                                           + 'load_dataset_seed{}'.format(load_dataset_seed)
                                           )
    cache_name1 = os.path.join('cache', (args.extra_datasets + '_lattice' + '_only_train:{}' +
                                         '_trainClip{}' + '_norm_num{}'
                                         + 'char_min_freq{}' + 'bigram_min_freq{}' + 'word_min_freq{}' + 'only_train_min_freq{}'
                                         + 'number_norm{}' + 'lexicon_{}' + 'load_dataset_seed{}')
                               .format(args.only_lexicon_in_train,
                                       args.train_clip, args.number_normalized, args.char_min_freq,
                                       args.bigram_min_freq, args.word_min_freq, args.only_train_min_freq,
                                       args.number_normalized, args.lexicon_name, load_dataset_seed))

    datasets, vocabs, embeddings = load_test(outdatasetPath, yangjie_rich_pretrain_unigram_path,
                                             yangjie_rich_pretrain_bigram_path,
                                             _refresh=refresh_data, index_token=False,
                                             train_clip=args.train_clip,
                                             _cache_fp=raw_dataset_cache_name1,
                                             char_min_freq=args.char_min_freq,
                                             bigram_min_freq=args.bigram_min_freq,
                                             only_train_min_freq=args.only_train_min_freq
                                             )
    datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(datasets, vocabs, embeddings,
                                                                  w_list, yangjie_rich_pretrain_word_path,
                                                                  _refresh=refresh_data, _cache_fp=cache_name1,
                                                                  only_lexicon_in_train=args.only_lexicon_in_train,
                                                                  word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                                  number_normalized=args.number_normalized,
                                                                  lattice_min_freq=args.lattice_min_freq,
                                                                  only_train_min_freq=args.only_train_min_freq)
    text = datasets  # 文本
    print(text['test'])
    CharacterToWord = CTW()
    tokenizer = ChineseTokenizer(cant_word_location)
    save_path = os.path.join(rootPth, "datasets/pickle_data")
    file_name = f"{args.dataset if args.extra_datasets == 'None' else args.extra_datasets}.pkl"
    res = {"tokenize": [], "wordVector": [], "fastIndexWord": {}}
    # write_to_pickle(os.path.join(save_path, file_name), res)
    suc = 0
    fai = 0
    for i in tqdm.tqdm(range(len(text['test'])), desc="将字向量转化为词向量"):
        sentence = text['test'][i:i + 1]
        sentence.set_target("target")
        sentence.set_input("bigrams")
        sentence.set_input("seq_len")
        sentence.set_input("lex_num")
        sentence.set_input("target")
        # sentence.print_field_meta()
        try:
            test_label_list = predictor.predict(sentence)  # 预测结果
        except Exception as e:
            fai += 1
            continue
        suc += 1
        test_raw_char = sentence['raw_chars']  # 原始文字
        sentence = ""
        for i in test_raw_char[0]:
            sentence += i
        mect4cner_out_vector = test_label_list['final_output']

        tokenize = tokenizer.tokenize(sentence)
        wordVector = CharacterToWord.run(mect4cner_out_vector, tokenize['wordGroupsID'])
        res['tokenize'].append(tokenize)
        res['wordVector'].append(wordVector)
        for Index in range(len(res['tokenize'][-1]['wordCutResult'])):
            word = res['tokenize'][-1]['wordCutResult'][Index]
            Vec = res['wordVector'][-1][Index]
            if word not in res['fastIndexWord']:
                res['fastIndexWord'][word] = Vec.unsqueeze(0)
            else:
                res['fastIndexWord'][word] = torch.cat((res['fastIndexWord'][word], Vec.unsqueeze(0)), dim=0)
    print(f"suc rate:{100 * suc / (suc + fai)}%")
    print(f"successfully save to {os.path.join(save_path, file_name)}")
    write_to_pickle(os.path.join(save_path, file_name), res)
    return res


if __name__ == '__main__':
    preprocess(1, 1)
