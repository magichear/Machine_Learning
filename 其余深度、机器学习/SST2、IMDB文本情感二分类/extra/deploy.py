import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import torch
import string
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from flask import Flask, request, jsonify

MODEL_PATH = r"D:\Study_Work\Electronic_data\CS\AAAUniversity\sdxxylysj\Lab2\model\bert-base-uncased"
MODEL_E2E_NAME = r"D:\Study_Work\Electronic_data\CS\AAAUniversity\sdxxylysj\Lab2\bert-base-uncased-e2e.bin"
MODEL_PRE_NAME = r"../model/bert-base-uncased-mlm.bin"
MODEL_PRE_PATH = r"../model/bert-base-uncased-mlm"
NUM_LABELS = 2


class BertClassifier(nn.Module):
    def __init__(self, mlm=False):
        super(BertClassifier, self).__init__()
        model_path = MODEL_PATH if not mlm else MODEL_PRE_PATH
        self.bert = BertModel.from_pretrained(model_path)

        self.dropout = nn.Dropout(0.6)
        self.classifier = nn.Linear(self.bert.config.hidden_size, NUM_LABELS)  # 分类层

    def forward(self, input_ids, attention_mask):
        # BERT 编码器
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask  # 区别有效区域与填充区域
        )
        # 基于 [CLS] token 的隐藏状态
        cls_output = outputs.pooler_output

        # Dropout + 分类层
        logits = self.classifier(cls_output)
        return logits


class BertSentence:
    def __init__(self, model_path, tokenizer_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model = BertClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("[SERVER] 模型初始化完成")

    def process(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]  # 如果是单条文本，转为列表
        sentences = [self.__clean_text(sentence) for sentence in sentences]
        print(f"[SERVER] 队列长度: {len(sentences)}")
        return self.__classify(sentences)

    def __classify(self, sentences):
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        inputs.pop("token_type_ids", None)  # 去除 token_type_ids
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # 把预处理数据传入模型
        with torch.no_grad():
            outputs = self.model(**inputs)
            labels = torch.argmax(outputs, dim=1)

        # 返回每条句子的分类结果
        return [label.item() for label in labels]

    def __clean_text(self, text):
        text = re.sub(r"<.*?>", "", text)  # 去除HTML标签
        text = text.translate(str.maketrans("", "", string.punctuation))  # 去除标点符号
        text = text.lower()  # 转为小写（uncased模型需要）
        text = re.sub(r"\s+", " ", text).strip()  # 去除多余空格
        return text


app = Flask(__name__)
print("加载模型和分词器...")
classifier = BertSentence(MODEL_E2E_NAME, MODEL_PATH)


@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.json
        sentences = data.get("sentences", [])
        if not sentences:
            return jsonify({"error": "No sentences provided"}), 400

        results = classifier.process(sentences)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
