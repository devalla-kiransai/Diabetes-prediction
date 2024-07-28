from flask import Flask, request, render_template, jsonify, redirect, url_for
import pickle
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# Define the home route
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


# Define the index route
@app.route("/index", methods=["GET"])
def index():
    return render_template("index.html")


# Define the route to handle redirection
@app.route("/redirect_to_index", methods=["POST"])
def redirect_to_index():
    return redirect(url_for("index"))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    data = [float(data[key]) for key in data]
    data = np.array(data).reshape(1, -1)

    # Scale the data using the loaded scaler
    data = scaler.transform(data)

    # Make prediction using the loaded model
    prediction = model.predict(data)

    # Convert prediction to int and prepare result
    output = int(prediction[0])
    result = "Positive for Diabetes" if output == 1 else "Negative for Diabetes"

    return jsonify(prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
