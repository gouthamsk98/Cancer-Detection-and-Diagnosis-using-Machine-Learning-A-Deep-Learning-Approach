# Cancer Detection and Diagnosis using Machine Learning: A Deep Learning Approach
This is a Flask web application that predicts whether an uploaded image is malignant or benign. The prediction is made using a pre-trained Keras model which was trained on a dataset of medical images.

### Requirements

-   Python 3.x
-   Flask
-   Tensorflow
-   Keras

### Installation

1.  Clone the repository

bashCopy code

`git clone https://github.com/username/flask-app.git`

1.  Install the dependencies

Copy code

`pip install -r requirements.txt`

### Usage

1.  Start the server by running the following command in your terminal:

Copy code

`python app.py`

1.  Open your web browser and go to `http://127.0.0.1:5000/`.

2.  Upload an image using the file upload form on the webpage and click "Predict".

3.  The web application will predict whether the uploaded image is malignant or benign.

### Model

The pre-trained Keras model used for image classification is located in the `models/weights.hdf5` file. The model is loaded using the `load_model()` function from the `keras.models` module.

### Credits

This web application was created by Goutham S Krishna and is based on the Keras Image Classification example.
