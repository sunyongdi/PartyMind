import torch
import jieba
import numpy as np
import os
import requests
import json

from jieba.analyse import extract_tags

from transformers import AutoTokenizer
from transformers import BertTokenizer

from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForSeq2SeqLM
from transformers import BertForSequenceClassification
from paddlenlp import Taskflow

from pycorrector.macbert.macbert_corrector import MacBertCorrector

from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from deepke.name_entity_re.standard.w2ner import Model as NerModel
from deepke.name_entity_re.standard.w2ner import dis2idx as ner_dis2idx
from deepke.name_entity_re.standard.w2ner import decode as ner_decode

from deepke.attribution_extraction.standard.tools import _handle_attribute_data
from deepke.attribution_extraction.standard.tools import _lm_serialize as ae_lm_serialize
from deepke.attribution_extraction.standard.models import LM as AE_LM

from deepke.relation_extraction.standard.models import LM as RE_LM
from deepke.relation_extraction.standard.tools import _handle_relation_data
from deepke.relation_extraction.standard.tools import _lm_serialize as re_lm_serialize


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
            keywords = [[model.wv.index_to_key[idx], float(sim_scores[idx])] for idx in top_k]
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
    def __init__(self, device) -> None:
        self.labels = ['会议', '事件', '文献', '组织机构']
        
        self.label2id = {label: ids for ids, label in enumerate(self.labels)}
        self.id2label = {ids: label for ids, label in enumerate(self.labels)}
        self.device = device
        self.model_path = '/root/sunyd/code/PartyMind/codes/text_classification/checkpoint/TextClassify_epoch2.pth'
        self.model = BertForSequenceClassification.from_pretrained('/root/sunyd/pretrained_models/bert-base-chinese/', num_labels=len(self.labels))
        self.model = self.load_model(self.model, self.device)
        self.tokenizer = BertTokenizer.from_pretrained('/root/sunyd/pretrained_models/bert-base-chinese/')
        self.model = self.model.to(self.device)
    
    def load_model(self, model, device):
        model_state = torch.load(self.model_path, map_location=device)
        model.load_state_dict(model_state)
        return model
    
    def encode(self, sentence):
        return self.tokenizer(sentence, max_length=512, pad_to_max_length=True, truncation = True, return_tensors='pt')
        
    
    def predict(self, sentence):
        inputs = self.encode(sentence)
        y_pred = self.model(inputs['input_ids'], token_type_ids=None)
        logits = y_pred.logits
        preds = torch.argmax(logits, dim=1)
        preds = preds.detach().numpy()[0]
        return self.id2label.get(preds, '事件')
    
    
class EntitySentiment:
    def __init__(self, device) -> None:
        self.schema = [{'人物': ['观点词', '情感倾向[正向，负向]']}, 
                       {'团体': ['观点词', '情感倾向[正向，负向]']} , 
                       {'会议': ['观点词', '情感倾向[正向，负向]']} , 
                       {'事件': ['观点词', '情感倾向[正向，负向]']} , 
                       {'文献': ['观点词', '情感倾向[正向，负向]']}  
                       ]
        self.ie = Taskflow('information_extraction', schema=self.schema, device_id=device)
        
        
    def entity_sentiment(self, sentence):
        res_list = []
        res = self.ie(sentence)[0]
        for entity_type in ['人物', '团体', '会议', '事件', '文献']:
            spo_list = res.get(entity_type)
            if spo_list is not None:
                res2 = self.serialize(entity_type, spo_list)
                if len(res2) == 0:
                    continue
                else:
                    res_list.extend(res2)
        return {'sentence': sentence, 'spo_list': res_list}
        
    def serialize(self, entity_type, people_spo_list):
        res = []
        for people_spo in people_spo_list:
            entity = (entity_type, people_spo['text'], people_spo['start'], people_spo['end'])
            relations = people_spo.get('relations')
            if relations is None:
                continue
            sentiment = relations.get('情感倾向[正向，负向]')[0]['text'] if relations.get('情感倾向[正向，负向]') else None
            opinion = relations.get('观点词')[0]['text'] if relations.get('观点词') else None
            if sentiment is None or opinion is None:
                continue
            res.append({
            'entity': entity,
            'sentiment': sentiment,
            'opinion': opinion
            })
        return res
        
    def predict(self, sentence):
        
        res = self.entity_sentiment(sentence)
        return res
    

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
        # tf_feature_names = tf_vectorizer.get_feature_names_out()
        tf_feature_names = tf_vectorizer.get_feature_names()
        return self.print_top_words(lda, tf, tf_feature_names, self.n_top_words)
    
    def predict(self):

        keywords = self.analysis()
        return keywords
    

class CorrectorZH:
    def __init__(self) -> None:
        self.model_path = '/root/sunyd/pretrained_models/shibing624/macbert4csc-base-chinese/'
        self.model = MacBertCorrector(self.model_path)

    def predict(self, sentence):
        err_info_global = []
        correct_query_global = ''
        for i in range(0, len(sentence), 100):
            correct_query, err_info = self.model.macbert_correct(sentence[i: i + 100])
            for single_info in err_info:
                err_info_global.append((single_info[0], single_info[1], single_info[2] + i, single_info[3] + i))
            correct_query_global += correct_query
        return correct_query_global, err_info_global


class NER:
    def __init__(self, device) -> None:
        self.device = device
        self.labels = ['<pad>', '<suc>', 'loc', 'per', 'org']
        self.label_num = len(self.labels)
        self.id2label = {ids: label for ids, label in enumerate(self.labels)}
        self.label2id = {label: ids for ids, label in enumerate(self.labels)}
        cfg = {
            'do_lower_case': True,
            'dist_emb_size': 20,
            'type_emb_size': 20,
            'lstm_hid_size': 512,
            'conv_hid_size': 96,
            'bert_hid_size': 768,
            'biaffine_size': 512,
            'ffnn_hid_size': 288,
            'dilation': [1, 2, 3],
            'emb_dropout': 0.5,
            'conv_dropout': 0.5,
            'out_dropout': 0.33,
            'use_bert_last_4_layers': True,
            'bert_name': '/root/sunyd/pretrained_models/bert-base-chinese/',
            'label_num': 5
        }
        config = type('Config', (), {})()
        for key in cfg.keys():
            config.__setattr__(key, cfg.get(key))
        
        self.model = NerModel(config)
        model = self.model.to(device)
        self.model.load_state_dict(torch.load('/root/sunyd/code/RelationExtraction/DeepKE/ner/standard/w2ner/output/pytorch_model.bin', map_location='cpu'))
        self.tokenizer = AutoTokenizer.from_pretrained('/root/sunyd/pretrained_models/bert-base-chinese/')
    
    
    def encode(self, sentence):
        length = len([word for word in sentence])
        tokens = [self.tokenizer.tokenize(word) for word in sentence]
        pieces = [piece for pieces in tokens for piece in pieces]
        bert_inputs = self.tokenizer.convert_tokens_to_ids(pieces)

        bert_inputs = np.array([self.tokenizer.cls_token_id] + bert_inputs + [self.tokenizer.sep_token_id])
        pieces2word = np.zeros((length, len(bert_inputs)), dtype=np.bool_)
        grid_mask2d = np.ones((length, length), dtype=np.bool_)
        dist_inputs = np.zeros((length, length), dtype=np.int64)
        sent_length = length

        if self.tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        for k in range(length):
            dist_inputs[k, :] += k
            dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if dist_inputs[i, j] < 0:
                    dist_inputs[i, j] = ner_dis2idx[-dist_inputs[i, j]] + 9
                else:
                    dist_inputs[i, j] = ner_dis2idx[dist_inputs[i, j]]
        dist_inputs[dist_inputs == 0] = 19
        
        return bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length
    
    def predict(self, sentence):
        bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length = self.encode(sentence)
        with torch.no_grad():
            result = []
            bert_inputs = torch.tensor([bert_inputs], dtype=torch.long).to(self.device)
            grid_mask2d = torch.tensor([grid_mask2d], dtype=torch.bool).to(self.device)
            dist_inputs = torch.tensor([dist_inputs], dtype=torch.long).to(self.device)
            pieces2word = torch.tensor([pieces2word], dtype=torch.bool).to(self.device)
            sent_length = torch.tensor([sent_length], dtype=torch.long).to(self.device)

            outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            length = sent_length
            grid_mask2d = grid_mask2d.clone()
            outputs = torch.argmax(outputs, -1)
            ent_c, ent_p, ent_r, decode_entities = ner_decode(outputs.cpu().numpy(), sentence, length.cpu().numpy())

            decode_entities = decode_entities[0]

            input_sentence = [word for word in sentence]
            for ner in decode_entities:

                ner_indexes, ner_label = ner
                entity = ''.join([input_sentence[ner_index] for ner_index in ner_indexes])
                entity_label = self.id2label[ner_label]
                result.append((entity, entity_label))
                
        return result


class AE:
    def __init__(self, device) -> None:
        self.device = device
        self.attribute_data = [{'attribute': 'None', 'index': '0'}, {'attribute': '民族', 'index': '1'}, {'attribute': '字', 'index': '2'}, {'attribute': '朝代', 'index': '3'}, {'attribute': '身高', 'index': '4'}, {'attribute': '创始人', 'index': '5'}, {'attribute': '上映时间', 'index': '6'}]
        cfg = {
            'model_name': 'lm',
            #lm_file: 'pretrained'
            'lm_file': '/root/sunyd/pretrained_models/bert-base-chinese/',
            'num_hidden_layers': 1,
            'type_rnn': 'LSTM',    # [RNN, GRU, LSTM]
            'input_size': 768,     # 这个值由bert得到
            'hidden_size': 100,    # 必须为偶数
            'num_layers': 1,
            'dropout': 0.3,
            'bidirectional': True,
            'last_layer_hn': True,
            'num_attributes': 7,
            'fp': '/root/sunyd/code/RelationExtraction/DeepKE/ae/checkpoints/2023-06-15_04-39-50/lm_epoch48.pth',
            'data_path': 'data/origin'
        }
        self.config = type('Config', (), {})()
        for key in cfg.keys():
            self.config.__setattr__(key, cfg.get(key))
        self.model = AE_LM(self.config)
        self.model.load(self.config.fp, device=device)
        self.model.to(device)
        self.model.eval()
    
    def _preprocess_data(self, data):
        atts = _handle_attribute_data(self.attribute_data)
        ae_lm_serialize(data, self.config)
        return data, atts


    def _get_predict_instance(self, sentence, entity, attribute_value):
        instance = dict()
        instance['sentence'] = sentence.strip()
        instance['entity'] = entity.strip()
        instance['attribute_value'] = attribute_value.strip()
        instance['entity_offset'] = sentence.find(entity)
        instance['attribute_value_offset'] = sentence.find(attribute_value)
        return instance
    
    def predict(self, sentence, entity, attribute_value):
        instance = self._get_predict_instance(sentence, entity, attribute_value)
        data = [instance]
        data, rels = self._preprocess_data(data)
        
        x = dict()
        x['word'], x['lens'] = torch.tensor([data[0]['token2idx'] + [0] * (512 - len(data[0]['token2idx']))]), torch.tensor([data[0]['seq_len']])
        for key in x.keys():
            x[key] = x[key].to(self.device)
            
        with torch.no_grad():
            y_pred = self.model(x)
            y_pred = torch.softmax(y_pred, dim=-1)[0]
            prob = y_pred.max().item()
            prob_att = list(rels.keys())[y_pred.argmax().item()]
        
        return data[0]['entity'], data[0]['attribute_value'], prob_att, prob
    
class RE:
    def __init__(self, device) -> None:
        self.device = device
        cfg = {
            'model_name': 'lm',
            'lm_file': '/root/sunyd/pretrained_models/bert-base-chinese/',
            'num_hidden_layers': 1,
            'type_rnn': 'LSTM',    # [RNN, GRU, LSTM]
            'input_size': 768,     # 这个值由bert得到
            'hidden_size': 100,    # 必须为偶数
            'num_layers': 1,
            'dropout': 0.3,
            'bidirectional': True,
            'last_layer_hn': True,
            'num_relations': 11,
            'fp': '/root/sunyd/code/RelationExtraction/DeepKE/re/checkpoints/2023-06-15_13-44-01/lm_epoch50.pth',
        }
        self.config = type('Config', (), {})()
        for key in cfg.keys():
            self.config.__setattr__(key, cfg.get(key))
        self.relation_data =  [{'head_type': 'None', 'tail_type': 'None', 'relation': 'None', 'index': '0'}, {'head_type': '影视作品', 'tail_type': '人物', 'relation': '导演', 'index': '1'}, {'head_type': '人物', 'tail_type': '国家', 'relation': '国籍', 'index': '2'}, {'head_type': '人物', 'tail_type': '地点', 'relation': '祖籍', 'index': '3'}, {'head_type': '电视综艺', 'tail_type': '人物', 'relation': '主持人', 'index': '4'}, {'head_type': '人物', 'tail_type': '地点', 'relation': '出生地', 'index': '5'}, {'head_type': '景点', 'tail_type': '城市', 'relation': '所在城市', 'index': '6'}, {'head_type': '歌曲', 'tail_type': '音乐专辑', 'relation': '所属专辑', 'index': '7'}, {'head_type': '网络小说', 'tail_type': '网站', 'relation': '连载网站', 'index': '8'}, {'head_type': '影视作品', 'tail_type': '企业', 'relation': '出品公司', 'index': '9'}, {'head_type': '人物', 'tail_type': '学校', 'relation': '毕业院校', 'index': '10'}]
        self.model = RE_LM(self.config)
        self.model.load(self.config.fp, device=device)
        self.model.to(device)
        self.model.eval()
        pass
    
    def _preprocess_data(self, data):
        rels = _handle_relation_data(self.relation_data)
        re_lm_serialize(data, self.config)

        return data, rels


    def _get_predict_instance(self, sentence, head, tail, head_type, tail_type):
        instance = dict()
        instance['sentence'] = sentence.strip()
        instance['head'] = head.strip()
        instance['tail'] = tail.strip()
        if head_type.strip() == '' or tail_type.strip() == '':
            self.config.replace_entity_with_type = False
            instance['head_type'] = 'None'
            instance['tail_type'] = 'None'
        else:
            instance['head_type'] = head_type.strip()
            instance['tail_type'] = tail_type.strip()
        return instance
    
    def predict(self, sentence, head, tail, head_type, tail_type):
        # get predict instance
        instance = self._get_predict_instance(sentence, head, tail, head_type, tail_type)
        data = [instance]

        # preprocess data
        data, rels = self._preprocess_data(data)
        x = dict()
        x['word'], x['lens'] = torch.tensor([data[0]['token2idx'] + [0] * (512 - len(data[0]['token2idx']))]), torch.tensor([data[0]['seq_len']])
        for key in x.keys():
            x[key] = x[key].to(self.device)

        with torch.no_grad():
            y_pred = self.model(x)
            y_pred = torch.softmax(y_pred, dim=-1)[0]
            prob = y_pred.max().item()
            prob_rel = list(rels.keys())[y_pred.argmax().item()]
            
        return data[0]['head'], data[0]['tail'], prob_rel, prob


class ExtractKeywords:
    def __init__(self) -> None:
        pass
    
    def predict(self, sentence, topK=5, withWeight=True, allowPOS=('n', 'nr', 'ns'), stopwords=None, userdict=None):
        # 加载停用词表
        # for stopword in stopwords:
        #     jieba.analyse.set_stop_words(stopword)
        
        # 添加用户辞典
        if userdict is not None:
            for word in userdict:
                jieba.add_word(word)
        
        # 提取关键词
        # TODO 添加通过词性筛选
        # keywords = jieba.analyse.extract_tags(sentence, topK=topK, withWeight=withWeight, allowPOS=allowPOS)
        keywords = jieba.analyse.extract_tags(sentence, topK=topK, withWeight=withWeight)
        
        # 过滤停用词
        if stopwords is not None:
            keywords = [[keyword, score]for keyword, score in keywords if keyword not in stopwords]
                
        
        return keywords
    
class IE:
    def __init__(self) -> None:
        pass
    
    def common_re(self, sentence):
        url = "http://192.168.4.188:18284/nlp/commonRe"
        payload = json.dumps({
            "sentence": f"{sentence}"
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload).json()
        return response
    
    def predict(self, sentence):
        res = self.common_re(sentence)
        table_data = [{
            'object_type': "",
            'predicate': item[1],
            'object': item[2],
            'subject_type': "",
            'subject': item[0],
        } for item in res[3][0]]
        return table_data
        
if __name__ == '__main__':
    pass
