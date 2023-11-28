from Utils.load_data import *
from Utils.paths import *
from Modules.CNNRadicalLevelEmbedding import CNNRadicalLevelEmbedding


def tiebaDatasets(refresh_data, train_clip, raw_dataset_cache_name, char_min_freq, bigram_min_freq, only_train_min_freq,
                  w_list, only_lexicon_in_train, number_normalized, lattice_min_freq, cache_name, radical_dropout,
                  char_dropout):
    datasets, vocabs, embeddings = load_tieba(tieba_path, yangjie_rich_pretrain_unigram_path,
                                              yangjie_rich_pretrain_bigram_path,
                                              _refresh=refresh_data, index_token=False, train_clip=train_clip,
                                              _cache_fp=raw_dataset_cache_name,
                                              char_min_freq=char_min_freq,
                                              bigram_min_freq=bigram_min_freq,
                                              only_train_min_freq=only_train_min_freq
                                              )

    datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(datasets, vocabs, embeddings,
                                                                  w_list, yangjie_rich_pretrain_word_path,
                                                                  _refresh=refresh_data, _cache_fp=cache_name,
                                                                  only_lexicon_in_train=only_lexicon_in_train,
                                                                  word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                                  number_normalized=number_normalized,
                                                                  lattice_min_freq=lattice_min_freq,
                                                                  only_train_min_freq=only_train_min_freq)
    max_seq_len = max(*map(lambda x: max(x['seq_len']), datasets.values()))
    embeddings['components'] = CNNRadicalLevelEmbedding(vocab=vocabs['lattice'], embed_size=50, char_emb_size=30,
                                                        filter_nums=[30],
                                                        kernel_sizes=[3], char_dropout=char_dropout,
                                                        dropout=radical_dropout, pool_method='max'
                                                        , include_word_start_end=False, min_char_freq=1)
    return datasets
