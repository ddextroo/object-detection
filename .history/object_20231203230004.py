from roboflow import Roboflow
rf = Roboflow(api_key="BNPmtVcTPzgRmg3Xxlq3")
project = rf.workspace().project("fruit-qbury")
model = project.version(3).model

# infer on a local image
print(model.predict("download (1).jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("download (1).jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())

from roboflow import Roboflow
import base64


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

predict, jsonData = predict_and_save_image("https://www.rd.com/wp-content/uploads/2022/12/GettyImages-1154073475-e1671134139339.jpg?resize=700%2C467")
print(jsonData)
