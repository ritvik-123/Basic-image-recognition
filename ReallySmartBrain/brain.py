from imageai.Prediction import ImagePrediction  # imports imageai module
import os  # imports os path
execution_path = os.getcwd()  # gives the current working directory
prediction = ImagePrediction()  # prediction takes the image prediction module
# sets the object mode or algo to mobile net v4 (which is downloaded in the same directory)
prediction.setModelTypeAsMobileNetV2()
# gives the path to the mobilenetv2 model
prediction.setModelPath(os.path.join(execution_path, "mobilenet_v2.h5"))
prediction.loadModel()  # loads the model for the prediction object
# predictions take the predicted image name, probabilty takes it's probabilty or closeness to the the result, result count tells the machine how many predictions to make
predictions, probabilities = prediction.predictImage(
    os.path.join(execution_path, "plane.jpg"), result_count=3)
# unpacks each prediction and probabilty
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
