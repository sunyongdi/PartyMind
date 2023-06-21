# coding: UTF-8
import torch
import json

from flask import Flask
from flask import render_template
from flask import jsonify
from flask import request

import argparse

from framework.log import my_log
from framework.model import LanguageIdentification
from framework.model import MachineTranslation
from framework.model import TextClustering
from framework.model import ChineseSegmenter
from framework.model import TextClassification
from framework.model import EntitySentiment
from framework.model import TopicKeywords
from framework.model import CorrectorZH
from framework.model import NER
from framework.model import AE
from framework.model import RE
from framework.model import ExtractKeywords
from framework.model import IE



# 接收参数
parser = argparse.ArgumentParser(description='接口名称')
parser.add_argument("--port", type=float, default=9902, help="端口")
parser.add_argument("--gpu_ids", type=str, default='-1',
                    help='gpu ids to use, -1 for cpu, 0 for gpu')

args = parser.parse_args()

# 配置日志
logger = my_log()
logger.info(f'args:{args.__dict__}')

# device
device = torch.device("cpu" if args.gpu_ids ==
                      '-1' else "cuda:" + args.gpu_ids)

logger.info(f'device:{device}')

# 预加载模型
lang_model = LanguageIdentification(device)
trans_model = MachineTranslation(device)
cluster_model = TextClustering()
segmenter_model = ChineseSegmenter()
classify_model = TextClassification(device)
sentiment_model = EntitySentiment(-1) # device_id -1:cpu
corrector_model = CorrectorZH()
ner_model = NER(device)
ae_model = AE(device)
re_model = RE(device)
keywords_model = ExtractKeywords()
ie_model = IE()

logger.info(f'显存占用:{torch.cuda.memory_allocated()}')

app = Flask(__name__, template_folder='/root/sunyd/code/PartyMind/dist/', static_folder='/root/sunyd/code/PartyMind/dist')


@app.route('/language_identifier', methods=['POST'])
def language_identifier():
    """
    语种识别接口
    """
    global lang_model, trans_model
    json_data = request.get_json()
    logger.info(f'json_data:{json_data}')
    sentence = json_data.get('sentence')
    logger.info(f'sentence:{sentence}')
    
    # 1.predict languages
    lang = lang_model.predict(sentence)
    logger.info(f'lang:{lang}')
    
    trans_sentence = sentence
    if lang == 'en':
        # 2.translation en to zh
        trans_sentence = trans_model.predict(sentence)
        logger.info(f'en_trans_sentence:{trans_sentence}')
    elif lang == 'zh':
        trans_sentence = sentence
    else:
        trans_sentence = '暂为支持该语种翻译'
    logger.info(f'显存占用2:{torch.cuda.memory_allocated()}')
    return jsonify({'sentence': sentence, 'result':{'trans_sentence': trans_sentence, 'lang': lang}})


@app.route('/text_cluster', methods=['POST'])
def text_cluster():
    """
    文本聚类接口
    """
    global cluster_model
    form_data = request.get_json()
    logger.info(f'form_data:{form_data}')
    # corpus = request.files['file']
    corpus = form_data.get('corpus')
    
    logger.info(f'file_type:{type(corpus)}')
    # 读取文件内容
    # corpus_docs = corpus.read().decode('utf-8').splitlines()
    corpus_docs = corpus.split('\n')
    logger.info(f'corpus_docs:{corpus_docs}')
    

    # 聚类的数量
    n_clusters = int(form_data.get('n_clusters'))
    logger.info(f'n_clusters:{n_clusters}')

    # 关键字的数量
    n_keywords = int(form_data.get('n_keywords'))
    logger.info(f'n_keywords:{n_keywords}')

    # 用户词表
    user_dict = form_data.get('user_dict')
    # user_dict = json.loads(user_dict)
    logger.info(f'user_dict:{user_dict}')
    
    # 停用词表
    stop_words = form_data.get('stop_words')
    # stop_words = json.loads(stop_words)
    logger.info(f'stop_words:{stop_words}')
    
    # 1.cluster
    cluster_res = cluster_model.predict(corpus_docs, n_clusters, n_keywords,  user_dict, stop_words)
    logger.info(f'cluster_res:{cluster_res}')

    logger.info(f'显存占用2:{torch.cuda.memory_allocated()}')
    return jsonify({'corpus': corpus_docs, 'user_dict': user_dict, 'stop_words': stop_words, 'result': cluster_res})


@app.route('/chinese_segmenter', methods=['POST'])
def chinese_segmenter():
    """
    中文分词接口
    """
    global segmenter_model
    json_data = request.get_json()
    logger.info(f'json_data:{json_data}')
    
    sentence = json_data.get('sentence')
    logger.info(f'sentence:{sentence}')
    
    # 用户词表
    user_dict = json_data.get('user_dict')
    logger.info(f'user_dict:{user_dict}')
    
    # 停用词表
    stop_words = json_data.get('stop_words')
    logger.info(f'stop_words:{stop_words}')

    # segmenter
    segmenter_res = segmenter_model.predict(sentence, user_dict, stop_words)
    logger.info(f'segmenter_res:{segmenter_res}')

    logger.info(f'显存占用2:{torch.cuda.memory_allocated()}')
    return jsonify({'sentence': sentence, 'user_dict': user_dict, 'stop_words': stop_words, 'result': segmenter_res})


@app.route('/cn_text_classification', methods=['POST'])
def cn_text_classification():
    """
    中文内容分类
    """
    global classify_model
    json_data = request.get_json()
    logger.info(f'json_data:{json_data}')
    
    sentence = json_data.get('sentence')
    logger.info(f'sentence:{sentence}')

    # classify
    classify_res = classify_model.predict(sentence)
    logger.info(f'classify_res:{classify_res}')

    logger.info(f'显存占用2:{torch.cuda.memory_allocated()}')
    return jsonify({'sentence': sentence,  'result': classify_res})


@app.route('/sentiment_analysis', methods=['POST'])
def sentiment_analysis():
    """
    观点分析(情感分析)
    """
    global sentiment_model
    json_data = request.get_json()
    logger.info(f'json_data:{json_data}')
    
    sentence = json_data.get('sentence')
    logger.info(f'sentence:{sentence}')

    # sentiment
    sentiment_res = sentiment_model.predict(sentence)
    logger.info(f'sentiment_res:{sentiment_res}')

    logger.info(f'显存占用2:{torch.cuda.memory_allocated()}')
    return jsonify(sentiment_res)


@app.route('/topic_keywords', methods=['POST'])
def topic_keywords():
    """
    主题词分析
    """
    form_data = request.get_json()
    logger.info(f'form_data:{form_data}')
    corpus = form_data.get('corpus')
    
    # 读取文件内容
    # corpus_docs = corpus.read().decode('utf-8').splitlines()
    corpus_docs = corpus.split('\n')
    logger.info(f'corpus_docs:{corpus_docs}')

    # 主题的数量
    n_components = int(form_data.get('n_components'))
    logger.info(f'n_components:{n_components}')

    # 关键字的数量
    n_top_words = int(form_data.get('n_top_words'))
    logger.info(f'n_top_words:{n_top_words}')
    
    topic_keywords = TopicKeywords(train_data=corpus_docs, n_components=n_components,
                                n_top_words=n_top_words)
    
    # topic keywords
    topic_keywords = TopicKeywords(train_data=corpus_docs, n_components=n_components,
                                n_top_words=n_top_words)
    keywords_res = topic_keywords.analysis()
    logger.info(f'keywords_res:{keywords_res}')
    res = []
    for _, v in keywords_res.items():
        topic = v['keywords'][0]
        keywords = v['keywords']
        corpus_docs = v['corpus']
        res.append({
            'topic': topic,
            'keywords': keywords,
            'corpus_docs': corpus_docs
        })
            

    logger.info(f'显存占用2:{torch.cuda.memory_allocated()}')
    return jsonify({'corpus': corpus_docs, 'result': res})


@app.route('/corrector', methods=['POST'])
def corrector():
    """
    常识校对
    """
    global corrector_model
    json_data = request.get_json()
    logger.info(f'json_data:{json_data}')
    
    sentence = json_data.get('sentence')
    logger.info(f'sentence:{sentence}')
    
    corrector_res = corrector_model.predict(sentence)
    logger.info(f'corrector_res:{corrector_res}')

    logger.info(f'显存占用2:{torch.cuda.memory_allocated()}')
    return jsonify(corrector_res)


@app.route('/ner', methods=['POST'])
def ner():
    """
    命名实体识别
    """
    global ner_model
    json_data = request.get_json()
    logger.info(f'json_data:{json_data}')
    
    sentence = json_data.get('sentence')
    logger.info(f'sentence:{sentence}')
    
    ner_res = ner_model.predict(sentence)
    logger.info(f'ner_res:{ner_res}')

    logger.info(f'显存占用2:{torch.cuda.memory_allocated()}')
    return jsonify(ner_res)

@app.route('/ae', methods=['POST'])
def ae():
    """
    实体属性识别
    """
    global ae_model
    json_data = request.get_json()
    logger.info(f'json_data:{json_data}')
    # sentence, entity, attribute_value
    sentence = json_data.get('sentence')
    logger.info(f'sentence:{sentence}')
    
    entity = json_data.get('entity')
    logger.info(f'entity:{entity}')
    
    attribute_value = json_data.get('attribute_value')
    logger.info(f'attribute_value:{attribute_value}')
    
    ae_res = ae_model.predict(sentence, entity, attribute_value)
    logger.info(f'ae_res:{ae_res}')

    logger.info(f'显存占用2:{torch.cuda.memory_allocated()}')
    return jsonify(ae_res)


@app.route('/re', methods=['POST'])
def re():
    """
    实体关系抽取
    """
    global re_model
    json_data = request.get_json()
    logger.info(f'json_data:{json_data}')
    # sentence, entity, attribute_value
    sentence = json_data.get('sentence')
    logger.info(f'sentence:{sentence}')
    
    head = json_data.get('head')
    logger.info(f'head:{head}')
    
    tail = json_data.get('tail')
    logger.info(f'tail:{tail}')
    
    head_type = json_data.get('head_type')
    logger.info(f'head_type:{head_type}')
    
    tail_type = json_data.get('tail_type')
    logger.info(f'tail_type:{tail_type}')
    
    re_res = re_model.predict(sentence, head, tail, head_type, tail_type)
    logger.info(f're_res:{re_res}')

    logger.info(f'显存占用2:{torch.cuda.memory_allocated()}')
    return jsonify(re_res)


@app.route('/disambiguation', methods=['POST'])
def disambiguation():
    """
    实体消歧
    """
    # global re_model
    # json_data = request.get_json()
    # logger.info(f'json_data:{json_data}')
    # # sentence, entity, attribute_value
    # sentence = json_data.get('sentence')
    # logger.info(f'sentence:{sentence}')
    
    # head = json_data.get('head')
    # logger.info(f'head:{head}')
    
    # tail = json_data.get('tail')
    # logger.info(f'tail:{tail}')
    
    # head_type = json_data.get('head_type')
    # logger.info(f'head_type:{head_type}')
    
    # tail_type = json_data.get('tail_type')
    # logger.info(f'tail_type:{tail_type}')
    
    # re_res = re_model.predict(sentence, head, tail, head_type, tail_type)
    # logger.info(f're_res:{re_res}')

    # logger.info(f'显存占用2:{torch.cuda.memory_allocated()}')
    sentence = '李娜曾是备受瞩目的一位网球选手，获得多非常多的奖项是中国网球界的骄傲，在她的运动员时期，身体健硕、皮肤深黑。'
    disambiguation_res = [
        {'mention': '李娜', 
         'candidate_entities': [
             {'subject_id': 1001, 'name': '李娜', 'entity_type': 'Person', 'desc': '李娜，1982年2月26日出生于湖北省武汉市，毕业于华中科技大学，中国女子网球运动员，2008年北京奥运会女子单打第四名，2011年法国网球公开赛、2014年澳大利亚网球公开赛女子单打冠军，亚洲第一位大满贯女子单打冠军。', 'score': 0.9862},
             {'subject_id': 2131, 'name': '李娜', 'entity_type': 'Person', 'desc': '李娜（1963年7月25日－），原名牛志红，出生于河南省郑州市，毕业于河南省戏曲学校，曾是中国大陆女歌手，出家后法名释昌圣。', 'score': 0.21101}
             ]
         },
        {'mention': '网球', 
         'candidate_entities': [
             {'subject_id': 1234, 'name': '网球', 'entity_type': 'Sports', 'desc': '网球是一项运动，常见的有一对一的单打和二对二的双打，对抗双方隔着球网，用球拍将网球击打至对方场地中，目标是令对方无法将球打回我方场地，是一种需要体力和脑力的运动，且需要非常好的技术。', 'score': 0.9862},
             {'subject_id': 2131, 'name': '网球', 'entity_type': 'Game', 'desc': '网球游戏', 'score': 0.21101},
             {'subject_id': 8771, 'name': '网球王子', 'entity_type': 'Anime', 'desc': '日本动漫', 'score': 0.21101}
             ]
         },
        ]
    return jsonify(disambiguation_res)


@app.route('/keyword_extraction', methods=['POST'])
def keyword_extraction():
    global keywords_model
    json_data = request.get_json()
    logger.info(f'json_data:{json_data}')
    # sentence, topK=5, withWeight=True, allowPOS=('n', 'nr', 'ns'), stopwords=None, userdict=None
    sentence = json_data.get('sentence')
    logger.info(f'sentence:{sentence}')
    
    topK = json_data.get('topK')
    logger.info(f'topK:{topK}')
    
    allowPOS = json_data.get('allowPOS')
    logger.info(f'allowPOS:{allowPOS}')
    
    stopwords = json_data.get('stopwords')
    logger.info(f'stopwords:{stopwords}')
    
    userdict = json_data.get('userdict')
    logger.info(f'userdict:{userdict}')
    
    withWeight = True
    
    res_keywords = keywords_model.predict(sentence, topK, withWeight, allowPOS, stopwords, userdict)
    logger.info(f'res_keywords:{res_keywords}')
    
    return jsonify(res_keywords)

@app.route('/information_extraction', methods=['POST'])
def information_extraction():
    global ie_model
    json_data = request.get_json()
    logger.info(f'json_data:{json_data}')
    sentence = json_data.get('sentence')
    logger.info(f'sentence:{sentence}')
    
    ie_res = ie_model.predict(sentence)
    logger.info(f'ie_res:{ie_res}')
    
    return jsonify(ie_res)



if __name__ == '__main__':
    app.run(debug=False, port=args.port, host='0.0.0.0')
    app.config['JSON_AS_ASCII'] = False
