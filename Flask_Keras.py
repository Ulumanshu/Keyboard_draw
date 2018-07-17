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
# from PIL import Image
# import io
import keras
from keras.models import load_model, model_from_json
import numpy as np
import flask
from flask import Flask, render_template, url_for, request, flash, redirect,\
    jsonify
import tensorflow as tf
from werkzeug.serving import run_simple
import base64
import re
from scipy.misc import imread, imresize

def load_ze_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)

    with open("model/kar_model_json.json") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    #load weights into new model
    model.load_weights("model/kar_model.h5")
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
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(image))
    image = imread('output.png', mode='L')
    image = imresize(image, target)
    print(image)
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

@app.route("/predict", methods=["GET","POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    results = None
    if flask.request.method == "POST":
        image = request.get_data()
        # preprocess the image and prepare it for classification
        image = prepare_image(image, target=(28, 28))
        # classify the input image and then initialize the list
        # of predictions to return to the client
        with graph.as_default():
            preds = model.predict(image)
            print(preds)
            results = np.argmax(preds, axis=1)
            print(np.argmax(preds))
            results = str(results[0])
            print(results)
            # loop over the results and add them to the list of
            # returned predictions
            # answer = str(results[0])
        # return display placeholder for html embed
    return results

# if this is the main thread of execution first load the model and
# then start the server

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    run_simple("localhost", 5000, app, use_reloader=True, use_debugger=True,
               use_evalex=True)
