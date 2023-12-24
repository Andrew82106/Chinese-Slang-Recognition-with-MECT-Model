import jieba
import pandas as pd


class ChineseTokenizer:
    def __init__(self, custom_words_file_path=None, test_pkl_word_list=None):
        """
        如果载入test_pkl_word_list，则需要从test.pkl中进行加载
        """
        if custom_words_file_path is None:
            return
        df = pd.read_excel(custom_words_file_path)
        try:
            custom_words = [word for word in df['word']] + [word for word in df['cant']]
            if test_pkl_word_list is not None:
                custom_words = list(set(custom_words + test_pkl_word_list))
            for word in custom_words:
                jieba.add_word(word)
            print("successfully imported custom word list")
        except Exception as e:
            print(f"fail to load custom word to jieba tokenizer with error {e}")

    @staticmethod
    def basicCut(sentence_):
        seg_list = jieba.cut(sentence_)
        return list(seg_list)

    def tokenize(self, sentence_):
        seg_list = self.basicCut(sentence_)
        word_groups = []
        word_cut_result = []
        start = 0
        for word in seg_list:
            end = start + len(word)
            word_groups.append(list(range(start, end)))
            word_cut_result.append(word)
            start = end
        word_groups_id = [[group[0], group[-1]] if group[0] != group[-1] else [group[0]] for group in word_groups]

        return {"wordGroupsID": word_groups_id, "wordCutResult": word_cut_result}


if __name__ == "__main__":
    # 测试类的方法
    tokenizer = ChineseTokenizer()
    sentence = "大家好我是玛丽，我有只小绵羊"
    output = tokenizer.tokenize(sentence)
    print(output)
