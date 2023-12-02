import jieba


class ChineseTokenizer:

    def basicCut(self, sentence):
        seg_list = jieba.cut(sentence)
        return list(seg_list)

    def tokenize(self, sentence):
        seg_list = self.basicCut(sentence)
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