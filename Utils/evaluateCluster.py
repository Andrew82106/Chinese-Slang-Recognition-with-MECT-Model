from fastNLP import Tester
from torch import nn
from fastNLP.io.loader import ConllLoader
from fastNLP.core.metrics import SpanFPreRecMetric, AccuracyMetric
from fastNLP import DataSet, Vocabulary

vocabs = {}


class EmptyModel(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def forward(self, preds):
        return {'pred': preds}


def convertRunningLog(resultFilePath="../Modules/runningLog.txt"):
    with open(resultFilePath, "r", encoding='utf-8') as f:
        result = f.read()
    newFilePath = resultFilePath.replace("runningLog.txt", "resultLog.bio")
    try:
        resultLst = eval(result)
        with open(newFilePath, "w", encoding='utf-8') as f:
            for pairs in resultLst:
                word = pairs[0]
                label = pairs[1]
                for i in range(len(word)):
                    f.write(f"{word[i]}\t{'O' if label else ('B-CANT' if i == 0 else 'I-CANT')}\n")
                if word == '。':
                    f.write("\n")
    except:
        resultLst = result.split("], [")
        with open(newFilePath, "w", encoding='utf-8') as f:
            for pairs in resultLst:
                pairs = pairs.replace("[[", "").replace("]]", "").replace("'", "").replace(" ", "")
                pairs = pairs.split(",")
                assert len(pairs) == 2, f"unexpected length:{len(pairs)} and content is {pairs}"
                word = pairs[0]
                try:
                    label = ~(eval(pairs[1]))
                except Exception as e:
                    print(pairs)
                    raise e
                for i in range(len(word)):
                    f.write(f"{word[i]}\t{'O' if not label else ('B-CANT' if i == 0 else 'I-CANT')}\n")
                if word == '。':
                    f.write("\n")
    return newFilePath


def evaluateDBScanMetric(resultFilePath="../Modules/runningLog.txt"):
    global vocabs
    new_Path = convertRunningLog(resultFilePath)
    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(
        "/Users/andrewlee/Desktop/Projects/Chinese-Slang-Recognition-with-MECT-Model/datasets/NER/test/input.bio")
    test_bundle = loader.load(new_Path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']
    datasets['train'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')

    datasetDict = {'chars': [], 'target0': [], 'seq_len': [], 'pred0': [], 'pred_sentence': []}
    for instance in datasets['test']:
        datasetDict['chars'].append(instance['chars'])
        datasetDict['target0'].append(instance['target'])
        datasetDict['seq_len'].append(instance['seq_len'])
    for instance in datasets['train']:
        datasetDict['pred0'].append(instance['target'])
        datasetDict['pred_sentence'].append(instance['chars'])
    datasets = DataSet(datasetDict)

    label_vocab = Vocabulary()
    label_vocab.from_dataset(datasets, field_name='target0')
    vocabs = {'label': label_vocab}

    datasets.apply_field(wordToID, field_name='target0', new_field_name='target')
    datasets.apply_field(wordToID, field_name='pred0', new_field_name='preds')

    datasets.set_target("target")
    datasets.set_input("preds")
    datasets.set_target("seq_len")

    f1_metric = SpanFPreRecMetric(vocabs['label'], pred='pred', target='target', seq_len='seq_len',
                                  encoding_type='bio')
    acc_metric = AccuracyMetric(pred='pred', target='target', seq_len='seq_len')
    acc_metric.set_metric_name('label_acc')
    metrics = [
        f1_metric,
        acc_metric
    ]

    model = EmptyModel(datasets)
    tester = Tester(data=datasets, model=model, metrics=metrics, batch_size=16)
    tester.test()


def wordToID(words):
    result = []
    for w, i in enumerate(words):
        result.append(vocabs['label'].word2idx[i])
    return result


if __name__ == '__main__':
    evaluateDBScanMetric()
    print("end")