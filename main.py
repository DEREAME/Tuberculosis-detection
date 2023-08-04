from flask import Flask, render_template, request
from keras.models import load_model
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='templates')

models = [
    {'name': 'DenseNet',
     'Accuracy': '28.78',
     'Precision': '19.20',
     'Recall': '27.60',
     'FMeasure': '22.59',
     'Placeholder':'DenseNet201 is a convolutional neural network architecture that was introduced by Huang et al. in 2017. DenseNet201 is an extension of the original DenseNet architecture that uses densely connected blocks to improve the flow of information through the network and reduce the number of parameters.In DenseNet201, each layer is connected to all previous layers in a dense block, allowing each layer to directly access the feature maps produced by all preceding layers. This helps to improve the flow of information through the network and reduces the vanishing gradient problem that can occur in very deep networks. '
     },
    {'name': 'MobileNet',
     'Accuracy': '78.04',
     'Precision': '79.53',
     'Recall': '76.87',
     'FMeasure': '75.74',
     'Placeholder':'MobileNetV2 is a convolutional neural network architecture that was introduced by Google in 2018 as an improvement to the original MobileNet architecture. It is designed to run efficiently on mobile and embedded devices with limited computational resources.MobileNetV2 uses a combination of depthwise separable convolutions and linear bottlenecks to reduce the computational cost and the number of parameters in the model. '
     },

    {'name': 'XceptionNet',
     'Accuracy': '95.12',
     'Precision': '94.79',
     'Recall': '94.72',
     'FMeasure': '94.74',
     'Placeholder':'XceptionNet is a convolutional neural network architecture that was introduced by François Chollet in 2016.XceptionNet uses depthwise separable convolutions instead of traditional convolutions, which reduces the computational cost and the number of parameters in the model. The depthwise separable convolution is split into two parts: a depthwise convolution that applies a separate filter to each channel of the input, and a pointwise convolution that combines the output of the depthwise convolution using a 1x1 convolution.'},
    {'name': 'NasNetLarge',
     'Accuracy': '96.58',
     'Precision': '96.32',
     'Recall': '96.33',
     'FMeasure': '96.32',
     'Placeholder':'NasNetLarge is a convolutional neural network architecture that was introduced by Zoph et al. in 2018. NasNet stands for Neural Architecture Search Network, as the architecture was discovered using an automated neural architecture search algorithmNasNetLarge uses a combination of normal cells and reduction cells to reduce the computational cost and the number of parameters in the network. The normal cells are used for feature extraction, while the reduction cells are used to reduce the spatial dimensions of the feature maps.'
     },
    {'name': 'InceptionNet',
     'Accuracy': '98.53',
     'Precision': '98.51',
     'Recall': '98.41',
     'FMeasure': '98.46',
     'Placeholder':'Inception-ResNet-v2 is a convolutional neural network architecture that was introduced by Szegedy et al. in 2016. It is a combination of the Inception architecture and the ResNet architecture.Inception-ResNet-v2 uses the Inception modules from the Inception architecture, which are designed to capture features at different scales. The Inception modules consist of multiple convolutional layers with different filter sizes, which are concatenated together to form a final output. This helps the network to capture both local and global features.'

     },
]


@app.route('/')
def index():
    return render_template("homepage.html", models=models)


@app.route('/model/<string:name>')
def model(name):
    for model in models:
        if model['name'] == name:
            return render_template("model.html", model=model)


@app.route('/testing',methods=["GET","POST"])
def test():
    if request.method == "GET":
        return render_template("testing.html")

    else:
        print("IN POST")
        file = request.files['file']
        fp = file.save('temp.jpg')
        img = cv2.imread('temp.jpg') #read images
        img = cv2.resize(img, (80,80)) #resize image
        im2arr = np.array(img)
        im2arr = im2arr.reshape(80,80,3)
        print(im2arr.shape)
        model = load_model("models/inception.hdf5")
        predictions = model.predict(im2arr.reshape(-1,80,80,3))
        predictions = np.argmax(predictions, axis=1)
        print(predictions)
        labels = ['COVID-19', 'Normal', 'TB']
        img = cv2.resize(im2arr, (1000,600)) # increase the image size
        cv2.putText(img, 'Prediction Output : '+labels[predictions[0]]+" Detected.", (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) # increase font size and reposition text
        plt.imsave('static/temp.jpg',img)
        return {"success":True}, 201
        



if __name__ == "__main__":
    app.run(debug=True)
