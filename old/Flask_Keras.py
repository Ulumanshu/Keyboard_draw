# USAGE
# Start the server:
# python Flask_Keras.py
# Submit a request via cURL:
# curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submit a a request via Python:
# python simple_request.py

# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask import Flask, render_template, url_for, request, flash, redirect,\
    jsonify
import io
import tensorflow as tf
from werkzeug.serving import run_simple
import json
import sys


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
app.debug = True
model = None
graph = None


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = ResNet50(weights="imagenet")
    global graph
    graph = tf.get_default_graph()


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

@app.route("/")
@app.route("/Home")
def home():
    # render homepage html from template
    return render_template('Home.html')

@app.route("/About", methods=["GET"])
def About():
    # render homepage html from template
    return render_template('About.html',  title="About")

@app.route("/predict", methods=["GET","POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    display = None

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            data = request.args.get('imgURI', 0, type=str)
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))
            # classify the input image and then initialize the list
            # of predictions to return to the client
            with graph.as_default():
                preds = model.predict(image)
                results = imagenet_utils.decode_predictions(preds)
                # loop over the results and add them to the list of
                # returned predictions
                for (imagenetID, label, prob) in results[0]:
                    display += ("{}: {:.4f}".format(label, prob)) + "/n"

        # otherwise, the request failed
        else:
            display = "Request failed"

    # return display placeholder for html embed
    return jsonify(display=display)

# if this is the main thread of execution first load the model and
# then start the server

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    run_simple("localhost", 5000, app, use_reloader=True, use_debugger=True,
               use_evalex=True)