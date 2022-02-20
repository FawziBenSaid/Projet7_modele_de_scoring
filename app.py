from flask import Flask, request, jsonify
import json


# load the pickle model
import joblib
model= joblib.load('model.pkl')



# create flask app
app = Flask(__name__)

@app.route('/')
def home():
    return '<h1>Hello the world</h1>'



@app.route("/predict", methods=['POST'])
def predict():

    json_ = request.get_json()
    data_json = json_['data']
    data_val = [list(data_json.values())]
    predict_val = model.predict(data_val)
    prediction = predict_val.tolist()

    return jsonify(Prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
