from transformers import BertModel, BertTokenizer
import torch

# 加载预训练的BERT模型和分词器
model_name = 'bert-base-uncased'  # 可以根据需要选择不同的预训练模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 输入句子
text = "你的输入句子"

# 分词和编码
input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

# 获取BERT模型的输出
with torch.no_grad():
    outputs = model(**input_ids)

# 获取每个词的词向量
last_hidden_states = outputs.last_hidden_state
word_vectors = last_hidden_states[0]  # 选择最后一层的隐藏状态作为词向量

# 打印每个词和对应的词向量
for i, word in enumerate(tokenizer.convert_ids_to_tokens(input_ids[0])):
    print(f"Word: {word}, Vector: {word_vectors[i]}")
