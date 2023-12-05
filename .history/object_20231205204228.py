from flask import Flask, request, jsonify
from roboflow import Roboflow
import base64
import os
from dotenv import load_dotenv


load_dotenv()
        
app = Flask(__name__)

api_key = os.getenv("API_KEY")
model = os.getenv("MODEL")
print(api_key)
print(model)
rf = Roboflow(api_key=api_key)
project = rf.workspace().project(model)
model = project.version(3).model

@app.route('/')
def hello():
    return "ProHealth Web Service"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_url = data.get('image_url')
    temp = 'predictData.jpg'

    if image_url is None:
        return jsonify({"error": "Missing 'image_url' parameter"}), 400

    try:
        prediction = model.predict(image_url, hosted=True)
        prediction.save(output_path=temp)
        jsonData = prediction.json()
        with open(temp, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        return jsonify({"encoded_image": encoded_image, "response": jsonData})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if temp:
            try:
                os.remove(temp)
            except OSError as e:
                pass

if __name__ == '__main__':
    app.run(debug=True)

