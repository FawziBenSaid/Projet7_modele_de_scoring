from flask import Flask, request, jsonify
import json


# load the pickle model
import os
import pickle
my_dir = os.path.dirname('model.pkl')
pickle_file_path = os.path.join(my_dir, 'model.pkl')
with open(pickle_file_path, 'rb') as pickle_file:
    model = pickle.load(pickle_file)
    





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
