import numpy as np 
from flask import Flask, request
from predict import make_prediction

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to my Flask API</h1>"
        "<h3>I hope you will find it interesting</h3>"
         "<p>Try using PostMan to make some inference</p>"
        "</body>"
        "</html>"
    )
    return body

@app.route("/predict", methods=["POST"])
def predict():
    data_json = request.get_json()
    
    sepal_length_cm = data_json["sepal_length_cm"]
    sepal_width_cm = data_json["sepal_width_cm"]
    petal_length_cm = data_json["petal_length_cm"]
    petal_width_cm = data_json["petal_width_cm"]

    data = np.array([[sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm]])
    predictions = make_prediction(data)
    
    return str(predictions)

if __name__ == "__main__":
    app.run()
