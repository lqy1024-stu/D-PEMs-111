from transformers import BertForSequenceClassification, BertTokenizer
import torch
import pandas as pd
import re

"""
读取 .csv 文件，并计算每段文本的毒性分数
"""

df = pd.read_csv("alpaca_generated_text.csv")  # 替换为你的文件路径
generated_texts = df["generated_text"].tolist()  # 转为列表4

# 离线加载
model = BertForSequenceClassification.from_pretrained("toxic-bert")
tokenizer = BertTokenizer.from_pretrained("toxic-bert")

# 使用方式类似 Detoxify
def predict_toxicity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return torch.sigmoid(outputs.logits).tolist()[0][0]

count = 0
sum_count = 0
sum_score = 0
for i in generated_texts:
    score = predict_toxicity(i)
    sum_score = sum_score + score
print("avg toxic score is: ", sum_score/len(generated_texts))
print(count / len(generated_texts))
