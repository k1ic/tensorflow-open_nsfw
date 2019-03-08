# -*- coding: utf-8 -*-
# test in python3.6 tensorflow-gpu1.8.0
# 启动命令: gunicorn -c ../conf/gunicorn_80.conf http_classify_nsfw:api
# test case: curl -s 'http://10.235.65.12/classify_from_url?url=https://pic4.zhimg.com/80/v2-d793ef70697509e624bc043457b67997_hd.jpg'
import sys
import os
import json
import re
import hashlib
import tensorflow as tf
import numpy as np

from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader
from flask import Flask, request
from urllib import request as uRequest

api = Flask(__name__)

@api.route('/classify_from_url', methods = ['GET'])
def do_classify_from_url():
    url = request.args.get('url', '')
    if re.match(r'^https?:/{2}\w.+$', url):
        #下载并暂存文件
        filePath = downImgToLocal(url)

        params = [__file__, '-m', 'data/open_nsfw-weights.npy', filePath]
        resClassify = doClassify(params)

        res = {}
        res['url'] = url
        res['score'] = resClassify

        return json.dumps(res)
    else:
        return "Not valid url %s" %url

def md5(str):
    import hashlib
    m = hashlib.md5()
    m.update(str.encode("utf-8"))
    return m.hexdigest()

def downImgToLocal(url):
    tmpFileName = '/tmp/' + md5(url) + '.jpg'

    if os.path.exists(tmpFileName):
        return tmpFileName

    with uRequest.urlopen(url) as web:
        # 为保险起见使用二进制写文件模式，防止编码错误
        with open(tmpFileName, 'wb') as outfile:
            outfile.write(web.read())
    return tmpFileName

def doClassify(argv):
    IMAGE_LOADER_TENSORFLOW = "tensorflow"
    IMAGE_LOADER_YAHOO = "yahoo"

    #构造参数
    args = type('', (), {})()
    args.input_type = 'tensor'
    args.model_weights = argv[2]
    args.image_loader = IMAGE_LOADER_YAHOO
    args.input_file = argv[3]

    model = OpenNsfwModel()

    config = tf.ConfigProto()
    #控制 GPU 显存使用
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        input_type = InputType[args.input_type.upper()]
        model.build(weights_path=args.model_weights, input_type=input_type)

        fn_load_image = None

        if input_type == InputType.TENSOR:
            if args.image_loader == IMAGE_LOADER_TENSORFLOW:
                fn_load_image = create_tensorflow_image_loader(tf.Session(graph=tf.Graph()))
            else:
                fn_load_image = create_yahoo_image_loader()
        elif input_type == InputType.BASE64_JPEG:
            import base64
            fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])

        sess.run(tf.global_variables_initializer())

        image = fn_load_image(args.input_file)

        predictions = \
            sess.run(model.predictions,
                     feed_dict={model.input: image})

    #释放占用的显存
    tf.reset_default_graph()

    #删除临时文件
    os.remove(args.input_file)

    #构造返回值
    res = {}
    res['SFW'] = str(predictions[0][0])
    res['NSFW'] = str(predictions[0][1])

    return res

if __name__ == "__main__":
    api.run(host='10.235.65.12', port=80)
