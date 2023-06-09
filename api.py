# coding: UTF-8
import torch
import json

from flask import Flask
from flask import jsonify
from flask import request

import argparse

from framework.log import my_log
from framework.model import LanguageIdentification
from framework.model import MachineTranslation
from framework.model import TextClustering


# 接收参数
parser = argparse.ArgumentParser(description='接口名称')
parser.add_argument("--port", type=float, default=9902, help="端口")
# parser.add_argument("--address", type=str,
#                     default='/api_template', help="接口地址")
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

# 加载模型
lang_model = LanguageIdentification(device)
trans_model = MachineTranslation(device)
cluster_model = TextClustering()

logger.info(f'显存占用:{torch.cuda.memory_allocated()}')

app = Flask(__name__)


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
    return jsonify({'sentence': sentence, 'trans_sentence': trans_sentence, 'lang': lang})


@app.route('/text_cluster', methods=['POST'])
def text_cluster():
    """
    文本聚类接口
    """
    global cluster_model
    form_data = request.form
    logger.info(f'form_data:{form_data}')
    corpus = request.files['corpus']
    
    # 读取文件内容
    corpus_docs = corpus.read().decode('utf-8').splitlines()
    logger.info(f'corpus_docs:{corpus_docs}')

    # 聚类的数量
    n_clusters = int(form_data.get('n_clusters'))
    logger.info(f'n_clusters:{n_clusters}')

    # 关键字的数量
    n_keywords = int(form_data.get('n_keywords'))
    logger.info(f'n_keywords:{n_keywords}')

    # 用户词表
    user_dict = form_data.get('user_dict')
    user_dict = json.loads(user_dict)
    logger.info(f'user_dict:{user_dict}')
    
    # 停用词表
    stop_words = form_data.get('stop_words')
    stop_words = json.loads(stop_words)
    logger.info(f'stop_words:{stop_words}')
    
    # 1.cluster
    cluster_res = cluster_model.predict(corpus_docs, n_clusters, n_keywords,  user_dict, stop_words)
    logger.info(f'cluster_res:{cluster_res}')

    logger.info(f'显存占用2:{torch.cuda.memory_allocated()}')
    return jsonify({'corpus': corpus_docs, 'user_dict': user_dict, 'stop_words': stop_words, 'data': cluster_res})


if __name__ == '__main__':
    app.run(debug=False, port=args.port, host='0.0.0.0')
    app.config['JSON_AS_ASCII'] = False
