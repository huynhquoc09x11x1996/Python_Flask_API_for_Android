import io

import PIL
from flask import Flask, request, jsonify
from svmutil import *
from io import StringIO, BytesIO
import cv2
import base64
from PIL import Image
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)


# root
@app.route("/")
def index():
    """
    this is a root dir of my server
    :return: str
    """
    return "This is root!!!!"


# GET
@app.route('/images/<image>')
def prdict_label_image(image):
    """
    this serves as a demo purpose
    :param user:
    :return: str
    """
    return "your input: %s" % image


# POST
@app.route('/api/post_some_data', methods=['POST'])
def get_text_prediction():
    """
    predicts requested text whether it is ham or spam
    :return: json
    """
    json = request.get_json()

    print("json: ", json)
    b64str = json['image']
    print("Base64 string: ", b64str)

    im = Image.open(io.BytesIO(base64.b64decode(b64str)))
    # path_save="./image.jpg"
    # im.save(path_save)


    if len(json['image']) == 0:
        return jsonify({'error': 'invalid input'})
    return recognize(im)  # android app will receive this string



def recognize(img):
    img=np.asarray(img)
    dict_classes = {
        1.0: 'May bay: Airplane',
        2.0: 'Bon Sai: Bon Sai',
        3.0: 'Rua Bien: Turtle',
        4.0: 'Thuyen buom: Sheet',
        5.0: 'Dong ho: Watch'}
    # load centers words
    centers_word = joblib.load("./centers_word1.pkl")
    # load model svm
    modelSVM = svm_load_model('./model1.pkl')
    # load model kmean
    modelKMean = joblib.load("./mbk1.pkl")
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    hist_new_image = np.bincount(modelKMean.predict(descriptors), minlength=modelKMean.n_clusters)
    x0, max_idx = gen_svm_nodearray(hist_new_image.tolist())
    label = libsvm.svm_predict(modelSVM, x0)
    return dict_classes[label]


# running web app in local machine
if __name__ == '__main__':
    app.run(debug=True,host='172.16.2.213', port=5000)
