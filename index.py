from flask import Flask, render_template, request
import os
import numpy as np

app=Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"

@app.route("/index")
def helloto():
    return "Hello World from index"

@app.route("/predict", methods=["GET", "POST"])
def predict_image():
    if request.method == "GET":
        return render_template("img_predict.html")
    else:
        UPLOAD_DIR = "data"
        # ImmutableMultiDict([('img', <FileStorage: 'dog.24.jpg' ('image/jpeg')>)])
        image_storage = request.files.get("img")
        image_storage.save(os.path.join(UPLOAD_DIR, image_storage.filename))
        result = _predictImage(os.path.join(UPLOAD_DIR, image_storage.filename))
        return render_template("img_predict.html", prediction_result = result)

model_json = "model/dogs_vs_cats.json"
model_Weight = "model/dogs_vs_cats.h5"
with open(model_json) as mj:
    modeljosn = mj.read()

import cv2
import  numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.vgg16 import preprocess_input
model=model_from_json(modeljosn)
model.load_weights(model_Weight)

def _predictImage(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    result = model.predict(image) # [0, 1] [1, 0]
    result=np.argmax(result)
    if result==1:
        return "DOG"
    else:
        return "CAT"

app.run(host="127.0.0.1", port=5003, debug=True)


""""
But two major API types are - 
1. GET - It means getting somthing from server 
2. POST - Sending something to Server  
"""
