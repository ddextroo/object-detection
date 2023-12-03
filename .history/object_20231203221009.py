import Roboflow
rf = Roboflow(api_key="BNPmtVcTPzgRmg3Xxlq3")
project = rf.workspace().project("fruit-qbury")
model = project.version(3).model

# infer on a local image
# print(model.predict("your_image.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("download (1).jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())