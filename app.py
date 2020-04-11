from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

from tensorflow import keras
import json
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
#tf_config = some_custom_config
sess = tf.Session() 
graph = tf.get_default_graph()
set_session(sess)
#global model, graph
# Model saved with Keras model.save()
#MODEL_PATH = 'models/weights.hdf5'
MODEL_PATH = 'C:/Users/goutham/project/flask-project/models/weights.hdf5'
# Load your trained model
model = load_model(MODEL_PATH)
# =============================================================================
# model._make_predict_function()          # Necessary
# =============================================================================
print('Model loaded. Start serving...')
graph = tf.get_default_graph()

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    
    #session = tf.InteractiveSession()
    #session = tf.Session(config=config)
    #with session.as_default():
    #        with session.graph.as_default():
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        img = image.load_img(img_path, target_size=(224, 224))
        x = np.reshape(img,[1,224,224,3])
        preds = model.predict_classes(x)
	
        return preds

            


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result= str(preds)              # Convert to string
        if result == '[0]':
            result='benign';
        else:
            result='malignant';
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

