from flask import Flask, request, jsonify
from roboflow import Roboflow
import base64
import requests
from PIL import Image
from io import BytesIO
import os

app = Flask(__name__)

rf = Roboflow(api_key="BNPmtVcTPzgRmg3Xxlq3")  # Replace with your Roboflow API key
project = rf.workspace().project("fruit-qbury")
model = project.version(3).model

def predict_and_save_image(image_url):
    model.predict(image_url, confidence=40, overlap=30).save("prediction.jpg")
    response = model.predict(image_url, confidence=40, overlap=30).json()
    
    # Save the predicted image
    prediction_image_path = "prediction.jpg"

    return prediction_image_path, response

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_url = data.get('image_url')

    if image_url is None:
        return jsonify({"error": "Missing 'image_url' parameter"}), 400

    try:
        prediction_image_path, jsonData = predict_and_save_image(image_url)
        encoded_image = encode_image(prediction_image_path)
        return jsonify({"encoded_image": encoded_image, "response": jsonData})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up: Remove the saved prediction image
        if prediction_image_path:
            try:
                os.remove(prediction_image_path)
            except OSError as e:
                pass

if __name__ == '__main__':
    app.run(debug=True)
