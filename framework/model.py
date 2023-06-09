import torch
import jieba
import numpy as np

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForSeq2SeqLM

from gensim.models import Word2Vec
from sklearn.cluster import KMeans


class LanguageIdentification:
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
        if stopwords and isinstance(stopwords, list):
            for word in stopwords:
                self.stopwords.add(word.strip())

        if userdict and isinstance(userdict, list):
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


if __name__ == '__main__':
    text_cluster = TextClustering()
    corpus = ['1919年五四运动爆发前后，由中国先进知识分子以“民主”和“科学”为旗帜，发起的反对封建思想、道德和文化，提倡进步思想和文化的思想启蒙运动。',
              '《新青年》出版为其兴起的标志。1915年9月15日，陈独秀在上海创办《青年杂志》，新文化运动拉开序幕。',
              '陈独秀在《青年杂志》创刊号上发表《敬告青年》，提出“国人而欲脱蒙昧时代，羞为浅化之民也，则急起直追，当以科学与人权并重”。',
              '第一次提出“科学”与“人权”，树起民主和科学两面大旗。1916年9月，《青年杂志》更名为《新青年》。1917年，蔡元培出任北京大学校长，他聘请陈独秀到北京大学任教，《新青年》杂志随之迁到北京。此后，北京大学的李大钊、胡适、鲁迅、钱玄同、刘半农等陆续参加到《新青年》编辑部中来。许多青年学生成为新文化运动的拥护者和参加者。',
              '由此，以北京大学为阵地、以《新青年》编辑部为核心，形成了新文化运动阵营，新文化运动迅猛发展。初期新文化运动的基本内容是：提倡民主和科学，反对专制和迷信盲从；提倡个性解放，反对封建礼教；提倡新文学，反对旧文学，实行文学革命。五四运动后发展为以传播马克思主义为中心的思想解放运动。',
              '新文化运动提出的两大基本口号：民主和科学，即“德先生”（Democracy）和“赛先生”（Science）。',
              '民主，一是指民主精神和民主思想；二是指资产阶级民主政治制度。科学，主要是指与封建迷信、蒙昧无知相对立的科学思想、科学精神以及认识和判断事物的科学方法，同时也指具体的科学技术、科学知识。新文化运动在民主和科学两面大旗的指引下，向封建主义思想文化发起前所未有的猛烈攻击，掀起了思想解放的浪潮。',
              '初期新文化运动的另一项主要内容是文学革命。',
              '1917年1月，胡适在《新青年》上发表《文学改良刍议》，较系统地提出文学改良的主张，提倡以白话文代替文言文，以白话文学为中国文学之正宗。',
              '陈独秀、鲁迅、钱玄同、刘半农、周作人等也积极响应，白话文逐渐取代文言文，成为文学的正宗。',
              '五四运动后，新文化运动中涌现出各种新思潮，领军人物逐渐分化。',
              '1916年天津市民抗议法国侵占老西开的反帝斗争。',
              '天津海光寺墙子河以南至八里台、佟楼一带俗称老西开。',
              '法国从1900年开始一直越界侵占老西开。1916年10月20日，悍然强占老西开。25日，天津各界市民8000余人举行集会，并提出六项抵制措施。',
              '11月12日，天津各界工人开始罢工，18日成立了1400余人的罢工团。',
              '工人还成立了工团和工团事务所，指挥罢工和领导游行示威。天津人民的正义斗争，得到全国人民的支持和声援，迫使法国侵略者不得不放弃完全侵占老西开的图谋。',
              '1917年3月（俄历2月），俄国爆发的二月革命推翻了沙皇专制统治，但国家政权却被地主和资产阶级代表的临时政府所把持。',
              '1917年11月7日（俄历10月25日），在列宁的领导下，彼得堡的工人群众发动武装起义，推翻了反动的资产阶级临时政府。8日，全俄工农兵苏维埃第二次代表大会通过列宁起草的《和平法令》和《土地法令》，并成立了第一届苏维埃政府，即人民委员会。列宁当选为人民委员会主席。',
              '随后，苏维埃政权在俄国各地相继建立。俄国十月社会主义革命取得胜利。俄国十月革命的胜利，建立了世界上第一个无产阶级专政的社会主义国家，极大地鼓舞了中国人民和中国的先进分子，对中国革命产生了巨大的影响。',
              '十月革命使中国人民认识到，帝国主义的力量不是不能战胜的，中国人民的反帝斗争不是孤立无援的，由此增强了斗争的勇气和信心。',
              '同时，“十月革命一声炮响，给我们送来了马克思列宁主义。',
              '”中国的先进分子由此认识到马克思主义对中国革命运动的指导作用，接受了马克思主义，树立了无产阶级世界观，并且在实践中得出向俄国革命学习、“走俄国人的路”的结论。']

    # print(text_cluster.predict(corpus))
    print(text_cluster.predict(corpus, None, None))
