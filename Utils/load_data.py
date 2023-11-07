import copy

from fastNLP import Vocabulary, DataSet
import os
from fastNLP import cache_results
from fastNLP.io.loader import ConllLoader
from functools import partial

from Modules.StaticEmbedding import StaticEmbedding
from Utils.utils import get_bigrams, get_skip_path, Trie


@cache_results(_cache_fp='cache/ontonotes4ner', _refresh=False)
def load_ontonotes4ner(path, char_embedding_path=None, bigram_embedding_path=None, index_token=True, train_clip=False,
                       char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0):
    train_path = os.path.join(path, 'train.char.bmes{}'.format('_clip' if train_clip else ''))
    dev_path = os.path.join(path, 'dev.char.bmes')
    test_path = os.path.join(path, 'test.char.bmes')

    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test']])
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                 field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                   field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                  field_name='target', new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['label'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
                                         min_freq=char_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings


@cache_results(_cache_fp='cache/resume_ner', _refresh=False)
def load_resume_ner(path, char_embedding_path=None, bigram_embedding_path=None, index_token=True,
                    char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0):
    train_path = os.path.join(path, 'train.char.bmes')
    dev_path = os.path.join(path, 'dev.char.bmes')
    test_path = os.path.join(path, 'test.char.bmes')

    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test']])
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                 field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                   field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                  field_name='target', new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
                                         min_freq=char_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings


@cache_results(_cache_fp='cache/msraner1', _refresh=False)
def load_msra_ner_1(path, char_embedding_path=None, bigram_embedding_path=None, index_token=True, train_clip=False,
                    char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0):
    if train_clip:
        train_path = os.path.join(path, 'train_dev.char.bmes_clip1')
        test_path = os.path.join(path, 'test.char.bmes_clip1')
    else:
        train_path = os.path.join(path, 'train_dev.char.bmes')
        test_path = os.path.join(path, 'test.char.bmes')

    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(train_path)
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    print(datasets.keys())
    # print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['test']])
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'], datasets['test'],
                                 field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['test'],
                                   field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['test'],
                                  field_name='target', new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['label'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
                                         min_freq=char_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings


@cache_results(_cache_fp='cache/weiboNER_uni+bi', _refresh=False)
def load_weibo_ner(path, unigram_embedding_path=None, bigram_embedding_path=None, index_token=True,
                   char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0, char_word_dropout=0.01, label='all'):
    loader = ConllLoader(['chars', 'target'])
    # bundle = loader.load(path)
    #
    # datasets = bundle.datasets

    # print(datasets['train'][:5])

    train_path = os.path.join(path, 'weiboNER_2nd_conll.train_deseg')
    dev_path = os.path.join(path, 'weiboNER_2nd_conll.dev_deseg')
    test_path = os.path.join(path, 'weiboNER_2nd_conll.test_deseg')

    paths = {}
    paths['train'] = train_path
    paths['dev'] = dev_path
    paths['test'] = test_path

    datasets = {}

    for k, v in paths.items():
        bundle = loader.load(v)
        datasets[k] = bundle.datasets['train']

    for k, v in datasets.items():
        print('{}:{}'.format(k, len(v)))
    # print(*list(datasets.keys()))
    vocabs = {}
    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()

    for k, v in datasets.items():
        # ignore the word segmentation tag
        if label == 'ne':
            v.apply_field(lambda x: [w if len(w) > 1 and w.split('.')[1] == 'NAM' else 'O' for w in x], 'target',
                          'target')
        if label == 'nm':
            v.apply_field(lambda x: [w if len(w) > 1 and w.split('.')[1] == 'NOM' else 'O' for w in x], 'target',
                          'target')
        v.apply_field(lambda x: [w[0] for w in x], 'chars', 'chars')
        v.apply_field(get_bigrams, 'chars', 'bigrams')

    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    print('label_vocab:{}\n{}'.format(len(label_vocab), label_vocab.idx2word))

    for k, v in datasets.items():
        # v.set_pad_val('target',-100)
        v.add_seq_len('chars', new_field_name='seq_len')

    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab

    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])
    if index_token:
        char_vocab.index_dataset(*list(datasets.values()), field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(*list(datasets.values()), field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(*list(datasets.values()), field_name='target', new_field_name='target')

    # for k,v in datasets.items():
    #     v.set_input('chars','bigrams','seq_len','target')
    #     v.set_target('target','seq_len')

    vocabs['bigram'] = bigram_vocab

    embeddings = {}

    if unigram_embedding_path is not None:
        unigram_embedding = StaticEmbedding(char_vocab, model_dir_or_name=unigram_embedding_path,
                                            word_dropout=char_word_dropout,
                                            min_freq=char_min_freq, only_train_min_freq=only_train_min_freq, )
        embeddings['char'] = unigram_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
                                           word_dropout=0.01,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings


@cache_results(_cache_fp='cache/demo', _refresh=True)
def load_demo(path, char_embedding_path=None, bigram_embedding_path=None, index_token=True, train_clip=False,
                    char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0):
    if train_clip:
        train_path = os.path.join(path, 'demo.bmes')
        test_path = os.path.join(path, 'demo.bmes')
    else:
        train_path = os.path.join(path, 'demo.bmes')
        test_path = os.path.join(path, 'demo.bmes')
    print(f"train_path:{train_path}")
    
    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(train_path)  # 这一句有invalid instance的情况出现
    test_bundle = loader.load(test_path)
    
    # fastNLP.io.loader.conll  ConllLoader：读取打标数据中的chars和target，调用load方法后返回的是一个包含句子和其标注的数据结构
    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']
    # 将train和test中的fastNLP.core.DataSet提取出来放入dataset中（主要是只有这部分是有用的）

    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    
    # apply_field方法：将DataSet中的每个instance中的名为 `field_name` 的field传给func，并获取它的返回值，放在名为`new_field_name`的field下。
    """
    get_bigrams函数：将输入的词语成对输出
    get_bigrams("hello")
    >>>['he', 'el', 'll', 'lo', 'o<end>']
    """
    # 因此这两句的意思就是将原来的datasets中的chars列的内容成对输出，结果放在同一instance的bigrams列下
    datasets['train'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')
    # add_seq_len方法：将使用len()直接对field_name中每个元素作用，将其结果作为seqence length, 并放入seq_len这个field。
    # 这两句就是统计句子长度用的
    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    print(datasets.keys())
    print(len(datasets['test']))
    print(len(datasets['train']))

    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['test']])
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    # from_dataset方法：使用dataset的对应field中词构建词典:
    # 因此上面这几句就是将datasets中的chars，bigrams，target三个field中的内容提取出来生成对应的Vocabulary对象char_vocab，bigram_vocab，label_vocab
    if index_token:
        char_vocab.index_dataset(datasets['train'], datasets['test'],
                                 field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['test'],
                                   field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['test'],
                                  field_name='target', new_field_name='target')
        # index_dataset：将DataSet中对应field的词转为数字，Example::

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['label'] = label_vocab  # 不太明白这句话的意义在哪里，这不是重复的嘛
    # 这里就将所有的Vocabulary对象集合起来放在vocabs这个字典里面，方便后续调用
    # tips：fastNLP.core.vocabulary对象：用于构建, 存储和使用 str 到 int 的一一映射
    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
                                         min_freq=char_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding
    # StaticEmbedding函数就简单理解为将词语转化为向量，维度为50维
    # 因此上面两部分就对单个字符char和双字符bigram进行向量化，然后放入embeddings这个字典中了
    return datasets, vocabs, embeddings


@cache_results(_cache_fp='cache/tieba', _refresh=True)
def load_tieba(path, char_embedding_path=None, bigram_embedding_path=None, index_token=True, train_clip=False,
                    char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0):
    if train_clip:
        train_path = os.path.join(path, 'tieba.bmes')
        test_path = os.path.join(path, 'tieba.bmes')
    else:
        train_path = os.path.join(path, 'tieba.bmes')
        test_path = os.path.join(path, 'tieba.bmes')
    print(f"train_path:{train_path}")
    
    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(train_path)  # 这一句有invalid instance的情况出现
    test_bundle = loader.load(test_path)
    
    # fastNLP.io.loader.conll  ConllLoader：读取打标数据中的chars和target，调用load方法后返回的是一个包含句子和其标注的数据结构
    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']
    # 将train和test中的fastNLP.core.DataSet提取出来放入dataset中（主要是只有这部分是有用的）

    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    
    # apply_field方法：将DataSet中的每个instance中的名为 `field_name` 的field传给func，并获取它的返回值，放在名为`new_field_name`的field下。
    """
    get_bigrams函数：将输入的词语成对输出
    get_bigrams("hello")
    >>>['he', 'el', 'll', 'lo', 'o<end>']
    """
    # 因此这两句的意思就是将原来的datasets中的chars列的内容成对输出，结果放在同一instance的bigrams列下
    datasets['train'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')
    # add_seq_len方法：将使用len()直接对field_name中每个元素作用，将其结果作为seqence length, 并放入seq_len这个field。
    # 这两句就是统计句子长度用的
    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    print(datasets.keys())
    print(len(datasets['test']))
    print(len(datasets['train']))

    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['test']])
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    # from_dataset方法：使用dataset的对应field中词构建词典:
    # 因此上面这几句就是将datasets中的chars，bigrams，target三个field中的内容提取出来生成对应的Vocabulary对象char_vocab，bigram_vocab，label_vocab
    if index_token:
        char_vocab.index_dataset(datasets['train'], datasets['test'],
                                 field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['test'],
                                   field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['test'],
                                  field_name='target', new_field_name='target')
        # index_dataset：将DataSet中对应field的词转为数字，Example::

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['label'] = label_vocab  # 不太明白这句话的意义在哪里，这不是重复的嘛
    # 这里就将所有的Vocabulary对象集合起来放在vocabs这个字典里面，方便后续调用
    # tips：fastNLP.core.vocabulary对象：用于构建, 存储和使用 str 到 int 的一一映射
    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
                                         min_freq=char_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding
    # StaticEmbedding函数就简单理解为将词语转化为向量，维度为50维
    # 因此上面两部分就对单个字符char和双字符bigram进行向量化，然后放入embeddings这个字典中了
    return datasets, vocabs, embeddings


@cache_results(_cache_fp='cache/load_yangjie_rich_pretrain_word_list', _refresh=False)
def load_yangjie_rich_pretrain_word_list(embedding_path, drop_characters=True):
    f = open(embedding_path, 'r')
    lines = f.readlines()
    w_list = []
    for line in lines:
        splited = line.strip().split(' ')
        w = splited[0]
        w_list.append(w)

    if drop_characters:
        w_list = list(filter(lambda x: len(x) != 1, w_list))

    return w_list


@cache_results(_cache_fp='need_to_defined_fp', _refresh=False)
def equip_chinese_ner_with_skip(datasets, vocabs, embeddings, w_list, word_embedding_path=None,
                                word_min_freq=1, only_train_min_freq=0):
    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)

    # for k,v in datasets.items():
    #     v.apply_field(partial(get_skip_path,w_trie=w_trie),'chars','skips')

    def skips2skips_l2r(chars, w_trie):
        '''

        :param lexicons: list[[int,int,str]]
        :return: skips_l2r
        '''
        # print(lexicons)
        # print('******')

        lexicons = get_skip_path(chars, w_trie=w_trie)

        # max_len = max(list(map(lambda x:max(x[:2]),lexicons)))+1 if len(lexicons) != 0 else 0

        result = [[] for _ in range(len(chars))]

        for lex in lexicons:
            s = lex[0]
            e = lex[1]
            w = lex[2]

            result[e].append([s, w])

        return result

    def skips2skips_r2l(chars, w_trie):
        '''

        :param lexicons: list[[int,int,str]]
        :return: skips_l2r
        '''
        # print(lexicons)
        # print('******')

        lexicons = get_skip_path(chars, w_trie=w_trie)

        # max_len = max(list(map(lambda x:max(x[:2]),lexicons)))+1 if len(lexicons) != 0 else 0

        result = [[] for _ in range(len(chars))]

        for lex in lexicons:
            s = lex[0]
            e = lex[1]
            w = lex[2]

            result[s].append([e, w])

        return result

    for k, v in datasets.items():
        v.apply_field(partial(skips2skips_l2r, w_trie=w_trie), 'chars', 'skips_l2r')

    for k, v in datasets.items():
        v.apply_field(partial(skips2skips_r2l, w_trie=w_trie), 'chars', 'skips_r2l')

    # print(v['skips_l2r'][0])
    word_vocab = Vocabulary()
    word_vocab.add_word_lst(w_list)
    vocabs['word'] = word_vocab
    for k, v in datasets.items():
        v.apply_field(lambda x: [list(map(lambda x: x[0], p)) for p in x], 'skips_l2r', 'skips_l2r_source')
        v.apply_field(lambda x: [list(map(lambda x: x[1], p)) for p in x], 'skips_l2r', 'skips_l2r_word')

    for k, v in datasets.items():
        v.apply_field(lambda x: [list(map(lambda x: x[0], p)) for p in x], 'skips_r2l', 'skips_r2l_source')
        v.apply_field(lambda x: [list(map(lambda x: x[1], p)) for p in x], 'skips_r2l', 'skips_r2l_word')

    for k, v in datasets.items():
        v.apply_field(lambda x: list(map(len, x)), 'skips_l2r_word', 'lexicon_count')
        v.apply_field(lambda x:
                      list(map(lambda y:
                               list(map(lambda z: word_vocab.to_index(z), y)), x)),
                      'skips_l2r_word', new_field_name='skips_l2r_word')

        v.apply_field(lambda x: list(map(len, x)), 'skips_r2l_word', 'lexicon_count_back')

        v.apply_field(lambda x:
                      list(map(lambda y:
                               list(map(lambda z: word_vocab.to_index(z), y)), x)),
                      'skips_r2l_word', new_field_name='skips_r2l_word')

    if word_embedding_path is not None:
        word_embedding = StaticEmbedding(word_vocab, word_embedding_path, word_dropout=0)
        embeddings['word'] = word_embedding

    vocabs['char'].index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                 field_name='chars', new_field_name='chars')
    vocabs['bigram'].index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                   field_name='bigrams', new_field_name='bigrams')
    vocabs['label'].index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                  field_name='target', new_field_name='target')

    return datasets, vocabs, embeddings


@cache_results(_cache_fp='need_to_defined_fp', _refresh=True)
def equip_chinese_ner_with_lexicon(datasets, vocabs, embeddings, w_list, word_embedding_path=None,
                                   only_lexicon_in_train=False, word_char_mix_embedding_path=None,
                                   number_normalized=False,
                                   lattice_min_freq=1, only_train_min_freq=0):
    from fastNLP.core import Vocabulary
    def normalize_char(inp):
        result = []
        for c in inp:
            if c.isdigit():
                result.append('0')
            else:
                result.append(c)

        return result

    def normalize_bigram(inp):
        result = []
        for bi in inp:
            tmp = bi
            if tmp[0].isdigit():
                tmp = '0' + tmp[:1]
            if tmp[1].isdigit():
                tmp = tmp[0] + '0'

            result.append(tmp)
        return result

    if number_normalized == 3:
        for k, v in datasets.items():
            v.apply_field(normalize_char, 'chars', 'chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                    no_create_entry_dataset=[datasets['dev'], datasets['test']])

        for k, v in datasets.items():
            v.apply_field(normalize_bigram, 'bigrams', 'bigrams')
        vocabs['bigram'] = Vocabulary()
        vocabs['bigram'].from_dataset(datasets['train'], field_name='bigrams',
                                      no_create_entry_dataset=[datasets['dev'], datasets['test']])

    if only_lexicon_in_train:
        print('已支持只加载在trian中出现过的词汇')

    def get_skip_path(chars, w_trie):
        sentence = ''.join(chars)
        result = w_trie.get_lexicon(sentence)
        # print(result)

        return result

    a = DataSet()
    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)

    if only_lexicon_in_train:
        lexicon_in_train = set()
        for s in datasets['train']['chars']:
            lexicon_in_s = w_trie.get_lexicon(s)
            for s, e, lexicon in lexicon_in_s:
                lexicon_in_train.add(''.join(lexicon))

        print('lexicon in train:{}'.format(len(lexicon_in_train)))
        print('i.e.: {}'.format(list(lexicon_in_train)[:10]))
        w_trie = Trie()
        for w in lexicon_in_train:
            w_trie.insert(w)

    for k, v in datasets.items():
        v.apply_field(partial(get_skip_path, w_trie=w_trie), 'chars', 'lexicons')
        v.apply_field(copy.copy, 'chars', 'raw_chars')
        v.add_seq_len('lexicons', 'lex_num')
        v.apply_field(lambda x: list(map(lambda y: y[0], x)), 'lexicons', 'lex_s')
        v.apply_field(lambda x: list(map(lambda y: y[1], x)), 'lexicons', 'lex_e')

    if number_normalized == 1:
        for k, v in datasets.items():
            v.apply_field(normalize_char, 'chars', 'chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                    no_create_entry_dataset=[datasets['dev'], datasets['test']])

    if number_normalized == 2:
        for k, v in datasets.items():
            v.apply_field(normalize_char, 'chars', 'chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                    no_create_entry_dataset=[datasets['dev'], datasets['test']])

        for k, v in datasets.items():
            v.apply_field(normalize_bigram, 'bigrams', 'bigrams')
        vocabs['bigram'] = Vocabulary()
        vocabs['bigram'].from_dataset(datasets['train'], field_name='bigrams',
                                      no_create_entry_dataset=[datasets['dev'], datasets['test']])

    def concat(ins):
        chars = ins['chars']
        lexicons = ins['lexicons']
        result = chars + list(map(lambda x: x[2], lexicons))
        return result

    def get_pos_s(ins):
        lex_s = ins['lex_s']
        seq_len = ins['seq_len']
        pos_s = list(range(seq_len)) + lex_s
        # lex_s是啥，为啥要加一个这个东西

        return pos_s

    def get_pos_e(ins):
        lex_e = ins['lex_e']
        seq_len = ins['seq_len']
        pos_e = list(range(seq_len)) + lex_e

        return pos_e

    for k, v in datasets.items():
        v.apply(concat, new_field_name='lattice')
        v.set_input('lattice')
        v.apply(get_pos_s, new_field_name='pos_s')
        v.apply(get_pos_e, new_field_name='pos_e')
        v.set_input('pos_s', 'pos_e')

    word_vocab = Vocabulary()
    word_vocab.add_word_lst(w_list)
    vocabs['word'] = word_vocab

    lattice_vocab = Vocabulary()
    lattice_vocab.from_dataset(datasets['train'], field_name='lattice',
                               no_create_entry_dataset=[v for k, v in datasets.items() if k != 'train'])
    vocabs['lattice'] = lattice_vocab

    if word_embedding_path is not None:
        word_embedding = StaticEmbedding(word_vocab, word_embedding_path, word_dropout=0)
        embeddings['word'] = word_embedding

    if word_char_mix_embedding_path is not None:
        lattice_embedding = StaticEmbedding(lattice_vocab, word_char_mix_embedding_path, word_dropout=0.01,
                                            min_freq=lattice_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['lattice'] = lattice_embedding

    vocabs['char'].index_dataset(*(datasets.values()),
                                 field_name='chars', new_field_name='chars')
    vocabs['bigram'].index_dataset(*(datasets.values()),
                                   field_name='bigrams', new_field_name='bigrams')
    vocabs['label'].index_dataset(*(datasets.values()),
                                  field_name='target', new_field_name='target')
    vocabs['lattice'].index_dataset(*(datasets.values()),
                                    field_name='lattice', new_field_name='lattice')

    return datasets, vocabs, embeddings
    """
    当这段代码被执行时，`datasets`、`vocabs` 和 `embeddings` 这三个变量会被填充以帮助构建和处理数据集、词汇表和嵌入向量。下面是它们的具体内容和作用：

1. `datasets`：它是一个字典，包含训练、开发和测试数据集。每个数据集都是一个由示例组成的数据集对象，包含不同的字段，比如字符、bigram、词汇、词典等。这些数据集对象可以用于训练模型和评估模型性能。

2. `vocabs`：这是一个字典，包含了不同类型的词汇表。在这个代码段中，`vocabs` 存储了字符、bigram、词汇和 lattice（可能是特定的词汇表类型）的词汇表。这些词汇表用于将文本数据中的字符、词汇和其他元素映射到唯一的索引，以便模型能够对其进行处理。

3. `embeddings`：这也是一个字典，存储了不同类型的嵌入向量。在这段代码中，它存储了与词汇表相关联的嵌入向量，如 `word_embedding` 和 `lattice_embedding`。这些嵌入向量用于将离散的字符或词汇转换为连续的向量表示，以便模型能够更好地处理和理解这些数据。

这些变量的填充内容是代码执行过程中生成的，它们的作用是帮助整个数据处理和模型训练过程能够顺利进行，同时为模型提供处理文本数据所需的必要信息。
    """
