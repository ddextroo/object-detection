import asyncio
from quart import Quart, request, jsonify
from roboflow import Roboflow
import base64
import requests
from PIL import Image
from io import BytesIO
import os
        
app = Quart(__name__)

rf = Roboflow(api_key="BNPmtVcTPzgRmg3Xxlq3")  # Replace with your Roboflow API key
project = rf.workspace().project("fruit-qbury")
model = project.version(3).model

@app.route('/predict', methods=['POST'])
async def predict():
    data = await request.json
    image_url = data.get('image_url')
    temp = 'predictData.jpg'

    if image_url is None:
        return jsonify({"error": "Missing 'image_url' parameter"}), 400

    try:
        prediction = await model.predict(image_url, hosted=True)
        prediction.save(output_path=temp)
        jsonData = prediction.json()
        with open(temp, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        return jsonify({"encoded_image": encoded_image, "response": jsonData})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up: Remove the saved prediction image
        if temp:
            try:
                os.remove(temp)
            except OSError as e:
                pass

if __name__ == '__main__':
    app.run(debug=True)

