# USAGE
# Start the server:
# python3 Flask_Keras_Multi.py
# Open web link and save letters to pc:D
from keras.models import load_model
import numpy as np
import flask
from flask import Flask, render_template, url_for, request, flash, redirect, jsonify
import tensorflow as tf
from werkzeug.serving import run_simple
import base64
import re
from scipy.misc import imread, imresize
import json
import os
import string
import pprint
from model.train_former import Train_Former as T

pp = pprint.PrettyPrinter(indent=4)


class Zemodel:
    @staticmethod
    def loadmodel(path):
        model = load_model(path)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        print("Model from {} file loaded".format(path))
        return model

    def __init__(self, path):
        self.model = self.loadmodel(path)
        self.graph = tf.get_default_graph()

    def zpredict(self, x):
        with self.graph.as_default():
            preds = self.model.predict(x)
            results = np.argmax(preds, axis=1)
            results = str(results)
            results = eval(results)
            results = results[0]
            return results

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
app.debug = True

def prepare_image(image, target):
    # resize the input image and preprocess it
    image = re.search(r'base64,(.*)', str(image)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(image))
    image = imread('output.png', mode='L')
    image = imresize(image, target)
    image = image.reshape(1, 42, 42, 1)
    # return the processed image
    return image

@app.route("/")
@app.route("/Home")
def home():
    # render homepage html from template
    return render_template('Home.html')

@app.route("/About")
def About():
    # render html from template
    with open('./model/TrFo_Self.json') as f:
        dataset = json.load(f)
#    html_json = json.dump(dataset, indent=4, sort_keys=True)
    return render_template('About.html', title="About", value=dataset)


@app.route("/postman", methods=["GET","POST"])
def postman():
    # refresh dataset nfo
    if flask.request.method == "POST":
        response = request.get_data()
        response = response.decode()
        if str(response) == "refresh_data":
            print('succes')
            count = T(save_dir="./static/Own_classes/save", train_dir="./static/Own_classes/train",)
            count.accountant()
        print(str(response))
    return 'succes'

@app.route("/save", methods=["GET","POST"])
def save():
    results = []
    filebase_dir = "./static/Own_classes/save"
    dir_lowercase = '/lowercase'
    dir_uppercase = '/uppercase'
    dir_numbers = '/numbers'
    if flask.request.method == "POST":
        response = request.get_data()
        alt_image = request.form["image"]
        image = re.search(r'base64,(.+)', str(alt_image)).group(1)
        c_class = re.search(r'&correct_class=(.)', str(response)).group(1)
        if len(re.search(r'&correct_class=(.+)', str(response)).group(1)) > 5:
            c_class = re.search(r"&correct_class=\w\w\w\w\w\w_(.)", str(response)).group(1)
        save_dir = []
        upercase = string.ascii_uppercase
        lowercase = string.ascii_lowercase
        digits = string.digits
        all_Valid = upercase + lowercase + digits
        if c_class in upercase:
            dir_end = "/letter_" + c_class
            e = filebase_dir + dir_uppercase + dir_end
            save_dir = e
        elif c_class in lowercase:
            dir_end = "/letter_" + c_class
            e = filebase_dir + dir_lowercase + dir_end
            save_dir = e
        elif c_class in digits:
            dir_end = "/number_" + c_class
            e = filebase_dir + dir_numbers + dir_end
            save_dir = e
        elif c_class not in all_Valid:
            error_msg = ["there is no such dir", c_class, "there is no such dir", c_class]
            json_res = jsonify(error_msg)
            return json_res
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        cnt = len(next(os.walk(save_dir))[2]) + 1
        fn_root = dir_end[1:] + "_" + str(cnt)
        fname = '%s.png' % fn_root
        pp.pprint(save_dir)
        with open(os.path.join(save_dir, fname), 'wb') as output:
            output.write(base64.b64decode(image))
        results.append(save_dir)
        results.append(fname)
    # return jsonified list for js parse, html display
    json_res = jsonify(results)
    return json_res

@app.route("/predict", methods=["GET", "POST"])
def predict():
    # if somebody accidently pushes a button "predict" href="/predict"
    results = None
    if flask.request.method == "POST":
        image = request.get_data()
        # preprocess the image and prepare it for classification
        image = prepare_image(image, target=(42, 42))
        # classify the input image and then initialize the list
        # of predictions to return to the client
        results = model_C.zpredict(image)
        # loop over the label_dict_tuples and
        # connect human meaning to prediction
        with open("./model/models_multi/labels_Classifajar.json") as f:
            labels_dict = json.load(f)
        for key, value in labels_dict.items():
            if value == results:
                results = key
        if results == "numbers":
            with open("./model/models_multi/labels_{}.json".format(results)) as f:
                labels_dict = json.load(f)
            results = model_n.zpredict(image)
            for key, value in labels_dict.items():
                if value == results:
                    results = key
        if results == "uppercase":
            with open("./model/models_multi/labels_{}.json".format(results)) as f:
                labels_dict = json.load(f)
            results = model_u.zpredict(image)
            for key, value in labels_dict.items():
                if value == results:
                    results = key
        if results == "lowercase":
            with open("./model/models_multi/labels_{}.json".format(results)) as f:
                labels_dict = json.load(f)
            results = model_l.zpredict(image)
            for key, value in labels_dict.items():
                if value == results:
                    results = key
        # return display placeholder for html embed
        return results

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    model_C = Zemodel("./model/models_multi/model_Classifajar.h5")
    model_u = Zemodel("./model/models_multi/model_uppercase.h5")
    model_l = Zemodel("./model/models_multi/model_lowercase.h5")
    model_n = Zemodel("./model/models_multi/model_numbers.h5")
#    run_simple("localhost", 5000, app, use_reloader=True, use_debugger=True, use_evalex=True)
    app.run(host='0.0.0.0', port=5000)
