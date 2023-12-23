from fastNLP import Tester
from torch import nn
from fastNLP.io.loader import ConllLoader
from fastNLP.core.metrics import SpanFPreRecMetric, AccuracyMetric
from fastNLP import DataSet, Vocabulary
from Utils.paths import *
vocabs = {}


class EmptyModel(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @staticmethod
    def forward(preds):
        return {'pred': preds}


def convertRunningLog(resultFilePath=clusterResult_path):
    with open(resultFilePath, "r", encoding='utf-8') as f:
        result = f.read()
    newFilePath = clusterResultBio_path
    try:
        resultLst = eval(result)
        Label_404 = 0
        with open(newFilePath, "w", encoding='utf-8') as f:
            for pairs in resultLst:
                word = pairs[0]
                label = pairs[1]
                if label == 404:
                    label = False
                    Label_404 += 1
                for i in range(len(word)):
                    f.write(f"{word[i]}\t{'O' if label else ('B-CANT' if i == 0 else 'I-CANT')}\n")
                if word == '。':
                    f.write("\n")
        print(f"[file format modify]\nSuccess modified file {clusterResult_path} to {newFilePath}")
        print(f"404词语共{Label_404}个，在{len(resultLst)}个词语中占比{100*Label_404/len(resultLst)}%")
    except Exception as e:
        raise Exception(f"{e}")
    return newFilePath


def evaluateDBScanMetric(resultFilePath=clusterResult_path):
    global vocabs
    new_Path = convertRunningLog(resultFilePath)
    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(Standard_Test_BIO)
    test_bundle = loader.load(new_Path)
    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']
    assert len(datasets['test']['chars']) == len(datasets['train']['chars']), f"生成用例和测试用例长度不等，请完全生成文本后再评测（长度：{len(datasets['test']['chars'])} {len(datasets['train']['chars'])}）"
    datasets['train'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')
    print(f'[testing result]\ncomparing file {Standard_Test_BIO} with {new_Path}')
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
