import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import os

current_file_path = os.path.abspath(__file__)
print('aa', current_file_path)
# 加载用户自定义词典
jieba.load_userdict('./base_dict/userdict.txt')

# 加载停用词表
stopwords = set()
with open('./base_dict/stopwords.txt', 'r') as f:
    for line in f:
        stopwords.add(line.strip())
# 分词函数
def cut_text(text):
    words = jieba.cut(text)
    return [word for word in words if word not in stopwords and len(word) > 1]

# 加载文本数据
corpus = []
with open('text_data.txt', 'r') as f:
    for line in f:
        corpus.append(line.strip())

# 训练词向量模型
sentences = [cut_text(text) for text in corpus]

model = Word2Vec(sentences, vector_size=100, min_count=5)

# 得到所有文本的词向量表示
X = []
for text in corpus:
    words = cut_text(text)
    vecs = []
    for word in words:
        if word in model.wv.key_to_index:
            vecs.append(model.wv[word])
    if len(vecs) == 0:
        vecs.append(np.zeros(100))
    X.append(np.mean(vecs, axis=0))

# 使用K-Means算法将文本聚类为5个簇
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(X)

# 对每个簇计算平均向量并生成主题词
res = []
for i in range(5):
    cluster_docs = [corpus[j] for j in range(len(corpus)) if labels[j] == i]
    cluster_vecs = [X[j] for j in range(len(X)) if labels[j] == i]
    avg_vec = np.mean(cluster_vecs, axis=0)
    sim_scores = model.wv.cosine_similarities(avg_vec, model.wv.vectors)
    top_k = np.argsort(sim_scores)[-5:]
    keywords = [model.wv.index_to_key[idx] for idx in top_k]
    res.append({
        "cluster": i,
        "keywords": keywords,
        'cluster_docs': cluster_docs
    })

print(res)