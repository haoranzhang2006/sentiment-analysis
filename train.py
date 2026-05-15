import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf

# 1. 加载数据
df = pd.read_csv("imdb_top_500.csv")
texts = df["review"].values
labels = df["sentiment"].map({"positive": 1, "negative": 0}).values

# 2. 加载 tiny_glove 词向量
with open("tiny_glove(1).json", "r") as f:
    glove = json.load(f)
embed_dim = len(glove["the"])  # 词向量维度，通常是100/200

# 3. 文本预处理函数
def preprocess(text):
    # 简单清洗 + 分词
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    return text.split()

# 4. 把评论转换成词向量序列
max_len = 100  # 每条评论固定长度为100个词
def text_to_vector(text):
    words = preprocess(text)
    vectors = []
    for word in words[:max_len]:
        if word in glove:
            vectors.append(glove[word])
        else:
            vectors.append(np.zeros(embed_dim))  # 未知词用零向量填充
    # 不足max_len的部分补零
    while len(vectors) < max_len:
        vectors.append(np.zeros(embed_dim))
    return np.array(vectors)

# 批量转换所有文本
X = np.array([text_to_vector(text) for text in texts])
y = np.array(labels)

# 5. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. 搭建神经网络（输入是词向量序列）
model = tf.keras.Sequential([
    # 可以直接用 Dense，也可以加简单的 Conv1D/LSTM，这里用最简单的
    tf.keras.layers.Flatten(input_shape=(max_len, embed_dim)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 7. 训练模型
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=16,
    verbose=1
)

# 8. 评估并保存模型
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ 测试集准确率: {acc:.4f}")

model.save("sentiment_model.h5")
print("✅ 模型已保存为 sentiment_model.h5")