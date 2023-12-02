import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
model = load_model('fish.h5')

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    file = request.files['file']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)
    img = image.load_img(image_path, target_size=(224, 224))  # load and resize image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    vgg16 = VGG16(include_top=False, weights='imagenet')
    bt_prediction = vgg16.predict(x)
    preds = model.predict(bt_prediction)
    fish = ['Catla', 'Common Carp', 'Mori', 'Rohu', 'Silver Carp']
    message = fish[np.argmax(preds)]
    return render_template('index.html', prediction_text=message, image_url=image_path)

if __name__ == "__main__":
    app.run(debug=True)