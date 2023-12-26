def extract_word_from_bio(r):
    with open(r, 'r', encoding='utf-8') as f:
        content = f.read()
    res_list = []
    content_list = content.split("\n")
    for word_pair in content_list:
        if '\t' not in word_pair:
            continue
        word, label = word_pair.split("\t")
        res_list.append(word)
    return res_list


def extract_sensitive_word_from_bio(r):
    """
    从.bio格式的文件中提取暗语词汇列表
    """
    with open(r, 'r', encoding='utf-8') as f:
        content = f.read()

    content_list = content.split("\n")
    res_list = []
    for word_pair in content_list:
        if '\t' not in word_pair:
            continue
        word, label = word_pair.split("\t")
        if label == 'B-CANT':
            res_list.append(word)
        elif label == 'I-CANT':
            res_list[-1] += word
    res_dict = {}
    for word in res_list:
        if word not in res_dict:
            res_dict[word] = 0
        res_dict[word] += 1
    return res_dict


def generate_compare(result_bio, input_bio):
    compare_word_dict = {}
    summary_word_dict = []
    for word in input_bio:
        if word not in result_bio:
            result_bio[word] = 0
        compare_word_dict[word] = input_bio[word] - result_bio[word]
        if compare_word_dict[word] > 5:
            print(f'{word}, {compare_word_dict[word]}')
            summary_word_dict.append(word)
    return summary_word_dict


if __name__ == '__main__':
    R = extract_sensitive_word_from_bio('/Users/andrewlee/Desktop/Projects/Chinese-Slang-Recognition-with-MECT-Model/clusterRes/Result.bio')
    I = extract_sensitive_word_from_bio('/Users/andrewlee/Desktop/Projects/Chinese-Slang-Recognition-with-MECT-Model/datasets/NER/test/input.bio')
    print(generate_compare(R, I))
    print(generate_compare(I, R))
