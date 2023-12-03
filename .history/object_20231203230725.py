from roboflow import Roboflow
import base64


rf = Roboflow(api_key="BNPmtVcTPzgRmg3Xxlq3")  # Replace with your Roboflow API key
project = rf.workspace().project("fruit-qbury")
model = project.version(3).model

prediction = model.predict("download (1).jpg")
prediction.plot()
prediction.save(output_path="predictData.jpg")
print(prediction.json())

