import os
import sys

if sys.platform == 'darwin':
    rootPth = "/Users/andrewlee/Desktop/Projects/Chinese-Slang-Recognition-with-MECT-Model"
else:
    rootPth = "B:\Chinese-Slang-Recognition-with-MECT-Model"
embeddings = os.path.join(rootPth, "datasets/embeddings")
charinfo = os.path.join(rootPth, "datasets/charinfo")
NER = os.path.join(rootPth, "datasets/NER")
vector = os.path.join(rootPth, "datasets/pickle_data")
cache_path = os.path.join(rootPth, "cache")

yangjie_rich_pretrain_unigram_path = os.path.join(embeddings, 'gigaword_chn.all.a2b.uni.ite50.vec')
yangjie_rich_pretrain_bigram_path = os.path.join(embeddings, 'gigaword_chn.all.a2b.bi.ite50.vec')
yangjie_rich_pretrain_word_path = os.path.join(embeddings, 'ctb.50d.vec')

# this path is for the output of preprocessing
yangjie_rich_pretrain_char_and_word_path = os.path.join(charinfo, 'yangjie_word_char_mix.txt')

# This is the path of the file with radicals
# radical_path = '/home/ws/data/char_info.txt'
radical_path = os.path.join(charinfo, 'chaizi-jt.txt')
radical_eng_path = os.path.join(charinfo, 'radicalEng.json')

ontonote4ner_cn_path = '/home/ws/data/OntoNote4NER'
msra_ner_cn_path = os.path.join(NER, 'MSRA_NER')
resume_ner_path = os.path.join(NER, 'resume_NER')
weibo_ner_path = os.path.join(NER, 'Weibo_NER')
demo_ner_path = os.path.join(NER, 'Demo_NER')
tieba_path = os.path.join(NER, 'tieba')
PKU_path = os.path.join(NER, 'PKU')
wiki_path = os.path.join(NER, 'wiki')
anwang_path = os.path.join(NER, 'anwang')
test_path = os.path.join(NER, 'test')

tieba_vector = os.path.join(vector, 'tieba.pkl')
weibo_vector = os.path.join(vector, 'weibo.pkl')
msra_vector = os.path.join(vector, 'msra.pkl')
PKU_vector = os.path.join(vector, 'PKU.pkl')
anwang_vector = os.path.join(vector, 'anwang.pkl')
wiki_vector = os.path.join(vector, 'wiki.pkl')

clusterLog_path = os.path.join(rootPth, "clusterLog", "clusterLog.txt")
if __name__ == '__main__':
    print(charinfo)
