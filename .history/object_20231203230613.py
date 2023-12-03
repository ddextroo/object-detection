from roboflow import Roboflow
import base64


rf = Roboflow(api_key="BNPmtVcTPzgRmg3Xxlq3")  # Replace with your Roboflow API key
project = rf.workspace().project("fruit-qbury")
model = project.version(3).model

prediction = model.predict("https://www.collinsdictionary.com/images/thumb/apple_158989157_250.jpg?version=5.0.34", hosted=True)
prediction.plot()
prediction.save(output_path="predictData.jpg")
print(prediction.json())

