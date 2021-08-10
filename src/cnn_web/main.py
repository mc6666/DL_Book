# Modification from https://github.com/marcotompitak/mnist-canvas
import tensorflow as tf
from flask import Flask, request
from flask_cors import CORS, cross_origin
import tensorflow as tf

from image_processing import preprocess
from utils import data_uri_to_cv2_img, value_invert

# Load in the saved neural network
model = tf.keras.models.load_model('./mnist_model.h5')

# Setting up the Flask app
app = Flask(__name__)

# Allow Cross-Origin Resource Sharing
cors = CORS(app)

@app.route('/')
def api_root():
    # Serve a canvas interface on /
    return app.send_static_file('interface.html')


@app.route('/post-data-url', methods=['POST'])
@cross_origin()
def api_predict_from_dataurl():

    # Read the image data from a base64 data URL
    imgstring = request.form.get('data')

    # Convert to OpenCV image
    img = preprocess(data_uri_to_cv2_img(imgstring))

    # Normalize values to 0-1, convert to white-on-black,
    # reshape for input layer
    data = value_invert(img/255).reshape((1,28,28,1))

    predictions = model.predict_classes(data)[0]

    print("Prediction requested! Returned " + str(predictions))

    # Return the prediction
    return str(predictions)


# Start flask app
if __name__ == '__main__':
    from os import environ
    app.run() #debug=False, port=environ.get("PORT", 5000), host='0.0.0.0')
