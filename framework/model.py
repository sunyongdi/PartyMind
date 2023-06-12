import torch
import jieba
import numpy as np

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForSeq2SeqLM

from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class LanguageIdentification:
    '''1.语种识别'''
    def __init__(self, device) -> None:
        self.device = device
        self.model_path = '/root/sunyd/pretrained_models/xlm-roberta-base-finetuned-language-identification'
        labels = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'it',
                  'ja', 'nl', 'pl', 'pt', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']

        self.languages2id = {lang: idx for idx, lang in enumerate(labels)}
        self.id2languages = {idx: lang for idx, lang in enumerate(labels)}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path)

        self.model = self.model.to(device)

    def encode(self, sentence):
        encode_token = self.tokenizer(sentence, return_tensors="pt")
        return encode_token

    def predict(self, sentence):
        self.model.eval()
        with torch.no_grad():
            encode_token = self.encode(sentence)
            encode_token = encode_token.to(self.device)
            pred = self.model(**encode_token).logits
            pre_idx = torch.argmax(pred).item()
            torch.cuda.empty_cache()

            return self.id2languages.get(pre_idx, 'en')


class MachineTranslation:
    '''机器翻译'''
    def __init__(self, device) -> None:
        self.device = device
        self.model_path = '/root/sunyd/pretrained_models/opus-mt-en-zh'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.model = self.model.to(self.device)

    def encode(self, sentence):
        encode_token = self.tokenizer(sentence, return_tensors='pt')
        return encode_token

    def predict(self, sentence):
        self.model.eval()
        with torch.no_grad():
            encode_token = self.encode(sentence)
            encode_token = encode_token.to(self.device)
            pred = self.model.generate(**encode_token)[0]
            zh_sentence = self.tokenizer.decode(pred, skip_special_tokens=True)
            torch.cuda.empty_cache()

            return zh_sentence


class TextClustering:
    '''文本聚类'''
    def __init__(self) -> None:
        self.userdict_path = '/root/sunyd/code/PartyMind/codes/text_clustering/base_dict/userdict.txt'
        self.stopwords_path = '/root/sunyd/code/PartyMind/codes/text_clustering/base_dict/stopwords.txt'
        
        # 加载用户自定义词典
        self.__load_userdict()
        
        # 加载停用词表
        self.stopwords = self.__add_stopwords()

    def __load_userdict(self):
        # 加载用户自定义词典
        jieba.load_userdict(self.userdict_path)

    def __add_stopwords(self):
        # 添加停用词表
        stopwords = set()
        with open(self.stopwords_path, 'r') as f:
            for line in f:
                stopwords.add(line.strip())
        return stopwords

    def __cut_text(self, text: str):
        # 分词函数
        words = jieba.cut(text)
        return [word for word in words if word not in self.stopwords and len(word) > 1]

    def predict(self, corpus, n_clusters=5, n_keywords=5, userdict=None, stopwords=None):
        if stopwords is not None and isinstance(stopwords, list):
            for word in stopwords:
                self.stopwords.add(word.strip())

        if userdict is not None and isinstance(userdict, list):
            for word in userdict:
                jieba.add_word(word.strip())

        # 训练词向量模型
        sentences = [self.__cut_text(text) for text in corpus]
        
        model = Word2Vec(sentences, vector_size=100, min_count=1)

        # 得到所有文本的词向量表示
        X = []
        for text in corpus:
            words = self.__cut_text(text)
            vecs = []
            for word in words:
                if word in model.wv.key_to_index:
                    vecs.append(model.wv[word])
            if len(vecs) == 0:
                vecs.append(np.zeros(100))
            X.append(np.mean(vecs, axis=0))

        # 使用K-Means算法将文本聚类为5个簇
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(X)

        # 对每个簇计算平均向量并生成主题词
        res = []
        for i in range(n_clusters):
            cluster_docs = [corpus[j]
                            for j in range(len(corpus)) if labels[j] == i]
            cluster_vecs = [X[j] for j in range(len(X)) if labels[j] == i]
            avg_vec = np.mean(cluster_vecs, axis=0)
            sim_scores = model.wv.cosine_similarities(
                avg_vec, model.wv.vectors)
            top_k = np.argsort(sim_scores)[-n_keywords:]
            keywords = [model.wv.index_to_key[idx] for idx in top_k]
            res.append({
                "cluster": i,
                "keywords": keywords,
                'cluster_docs': cluster_docs
            })

        return res
    

class ChineseSegmenter:
    '''中文分词'''
    def __init__(self):
        self.userdict_path = '/root/sunyd/code/PartyMind/codes/text_clustering/base_dict/userdict.txt'
        self.stopwords_path = '/root/sunyd/code/PartyMind/codes/text_clustering/base_dict/stopwords.txt'
        
        # 加载用户自定义词典
        self.__load_userdict()
        
        # 加载停用词表
        self.stopwords = self.__add_stopwords()

    def __load_userdict(self):
        # 加载用户自定义词典
        jieba.load_userdict(self.userdict_path)

    def __add_stopwords(self):
        # 添加停用词表
        stopwords = set()
        with open(self.stopwords_path, 'r') as f:
            for line in f:
                stopwords.add(line.strip())
        return stopwords
    
    
    def predict(self, text, userdict=None, stopwords=None):
        
        if stopwords is not None and isinstance(stopwords, list):
            for word in stopwords:
                self.stopwords.add(word.strip())

        if userdict is not None and isinstance(userdict, list):
            for word in userdict:
                jieba.add_word(word.strip())
                
        words = jieba.cut(text)
        return [word.strip() for word in words if word.strip() not in self.stopwords]


class TextClassification:
    '''知识分类'''
    def __init__(self) -> None:
        pass
    
    def predict(self, sentence):
        classify = [1,2,3,4]
        return 1
    
    
class SentimentAnalysis:
    '''观点分析'''
    def __init__(self) -> None:
        pass
    
    def predict(self, sentence):
        classify = [1,2,3,4]
        return 1
    

class TopicKeywords:
    """
    主题发现
    """
    def __init__(self, train_data, n_components=10, n_top_words=50, max_iter=50):
        """
        :param train_data: 训练数据
                      格式：   ["张三在中国移动工作", "你是谁？"]
        :param n_components:  主题数目
        :param n_top_words:  每个主题提取的主题词数目
        :param max_iter:  迭代次数
        """
        self.train_data = [" ".join(jieba.lcut(data)) for data in train_data]
        self.n_components = n_components
        self.n_top_words = n_top_words
        self.max_iter = max_iter

    def print_top_words(self, model, tf, feature_names, n_top_words):
        ret = {}
        for topic_idx, topic in enumerate(model.components_):
            key = "topic_{}".format(topic_idx)
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            ret[key] = {'keywords': top_words, 'corpus': []}
        
        doc_topic_dists = model.transform(tf)
        for i, doc in enumerate(self.train_data):
            topic_probs = doc_topic_dists[i]
            top_topic = topic_probs.argmax()
            key = "topic_{}".format(top_topic)
            ret[key]['corpus'].append(doc)
        return ret

    def analysis(self):
        tf_vectorizer = CountVectorizer()

        tf = tf_vectorizer.fit_transform(self.train_data)

        lda = LatentDirichletAllocation(n_components=self.n_components, max_iter=self.max_iter,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        lda.fit(tf)
        tf_feature_names = tf_vectorizer.get_feature_names_out()
        return self.print_top_words(lda, tf, tf_feature_names, self.n_top_words)
    
    def predict(self):

        keywords = self.analysis()
        return keywords
    



if __name__ == '__main__':
    pass
