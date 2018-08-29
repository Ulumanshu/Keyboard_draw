# USAGE
# Start the server:
# python Flask_Keras.py
# Submit a request via cURL:
# curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submit a a request via Python:
# python simple_request.py

# import the necessary packages
# from keras.applications import ResNet50
# from keras.preprocessing.image import img_to_array
# from keras.applications import imagenet_utils
from PIL import Image
# import io
#import keras
from keras.models import load_model
import numpy as np
import flask
from flask import Flask, render_template, url_for, request, flash, redirect,\
    jsonify
import tensorflow as tf
from werkzeug.serving import run_simple
import base64
import re
from scipy.misc import imread, imresize
import json
import os
import string
import pprint
pp = pprint.PrettyPrinter(indent=4)
def load_ze_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)

    # with open("old/kar_model2.json") as json_file:
    #     loaded_model_json = json_file.read()
    # model = model_from_json(loaded_model_json)
    #load weights into new model
    model = load_model("model/kar_model_balanced.h5")
    print("Loaded Model from disk")
    #compile and evaluate loaded model
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    graph = tf.get_default_graph()

    return model, graph


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
app.debug = True
global model, graph
model, graph = load_ze_model()


def prepare_image(image, target):

    # resize the input image and preprocess it
    image = re.search(r'base64,(.*)', str(image)).group(1)
    # pp.pprint("old " + image)
    # pp.pprint(len(image) % 4)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(image))
    image = imread('output.png', mode='L')
    image = imresize(image, target)
    #pp.pprint(image)
    image = image.reshape(1, 28, 28, 1)
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
    return render_template('About.html',  title="About")

@app.route("/save", methods=["GET","POST"])
def save():
    results = []
    subdir_lowercase = ['/home/wooden/Desktop/Pycharm_projects/Flask_Keras/static/Emnist_dir/Own_classes/lowercase/test', '/home/wooden/Desktop/Pycharm_projects/Flask_Keras/static/Emnist_dir/Own_classes/lowercase/train']
    subdir_uppercase = [
        '/home/wooden/Desktop/Pycharm_projects/Flask_Keras/static/Emnist_dir/Own_classes/uppercase/test',
        '/home/wooden/Desktop/Pycharm_projects/Flask_Keras/static/Emnist_dir/Own_classes/uppercase/train']
    subdir_numbers = [
        '/home/wooden/Desktop/Pycharm_projects/Flask_Keras/static/Emnist_dir/Own_classes/numbers/test',
        '/home/wooden/Desktop/Pycharm_projects/Flask_Keras/static/Emnist_dir/Own_classes/numbers/train']
    if flask.request.method == "POST":
        response = request.get_data()
        #pp.pprint(response)
        # pattern = re.compile(r"base64,?(.+)&correct_class=([a-zA-Z0-9]?)")
        # image = pattern.sub(r"\1", str(response))
        # c_class = pattern.sub(r"\2", str(response))
        # image = pattern.findall(str(response))
        # c_class = pattern.findall(str(response))
        # image = re.search(r'base64%2C(.+)&correct_class=(.)',
        #                   str(response)).group(1)
        alt_image = request.form["image"]
        image = re.search(r'base64,(.+)', str(alt_image)).group(1)
        c_class = re.search(r'&correct_class=(.)', str(response)).group(1)
        # c_class = re.search(r"[a-zA-Z1-9]", str(c_class))
        #c_class = request.form["correct_class"]
        #pp.pprint(c_class)
        # pp.pprint(alt_image)
        # pp.pprint("new " + image)
        # while len(image) % 4 != 0:
        #     image += "="
        # pp.pprint(len(image) % 4)
        #image = re.search(r'base64,(.*)', str(response)).group(1)
        #c_class = re.search(r'base64,(.*)', str(response)).group(1)
        save_dir =[]
        upercase = string.ascii_uppercase
        lowercase = string.ascii_lowercase
        digits = string.digits
        if c_class in upercase:
            for e in subdir_uppercase:
                e += "/letter_" + c_class
                save_dir.append(e)
        elif c_class in lowercase:
            for e in subdir_lowercase:
                e += "/letter_" + c_class
                save_dir.append(e)
        elif c_class in digits:
            for e in subdir_numbers:
                e += "/number_" + c_class
                save_dir.append(e)
        #pp.pprint(save_dir)
        for e in save_dir:
            if os.path.exists(e) == False:
                os.makedirs(e)
            cnt = len(next(os.walk(e))[2]) + 1
            fname = '%d.png' % cnt
            #pp.pprint(e)
            #pp.pprint(fname)
            #image = imread('output.png', mode='L')
            #pp.pprint(image)
            with open(os.path.join(e, fname), 'wb') as output:
                output.write(base64.b64decode(image))
            #image.save(fname)
            results.append(e)
            results.append(fname)
    print_str = ""
    for e in results:
        print_str += e + " "
    return print_str


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    results = None
    if flask.request.method == "POST":
        image = request.get_data()
        #pp.pprint(image)
        # preprocess the image and prepare it for classification
        image = prepare_image(image, target=(28, 28))
        # classify the input image and then initialize the list
        # of predictions to return to the client
        with graph.as_default():
            preds = model.predict(image)
            #pp.pprint(preds)
            results = np.argmax(preds, axis=1)
            results = str(results)
            results = eval(results)
            results = results[0]
            #pp.pprint(type(results))
            with open("./model/labels.json") as f:
                labels_dict = json.load(f)
                #pp.pprint(labels_dict)
            for key, value in labels_dict.items():
                if value == results:
                    results = key
            #pp.pprint(type(results))
            # loop over the results and add them to the list of
            # returned predictions
            # answer = str(results[0])
        # return display placeholder for html embed
    return results

# if this is the main thread of execution first load the model and
# then start the server

if __name__ == "__main__":
    pp.pprint(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    run_simple("localhost", 5000, app, use_reloader=True, use_debugger=True,
               use_evalex=True)
