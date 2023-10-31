import os

rootPth = "/home/MECT4CNER"
embeddings = os.path.join(rootPth, "datasets/embeddings")
charinfo = os.path.join(rootPth, "datasets/charinfo")
NER = os.path.join(rootPth, "datasets/NER")


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
resume_ner_path = '/home/ws/data/ResumeNER'
weibo_ner_path = os.path.join(NER, 'Weibo_NER')
demo_ner_path = os.path.join(NER, 'Demo_NER')

if __name__ == '__main__':
    print(charinfo)