import pickle
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# load the model
with open('model25.pickle', 'rb') as f:
    model = pickle.load(f)

# load the encoder
with open('encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)

@app.route('/')
def home():
    return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict():
    symptom_1 = request.form.get('Symptom_1')
    symptom_2 = request.form.get('Symptom_2')
    symptom_3 = request.form.get('Symptom_3')
    symptom_4 = request.form.get('Symptom_4')
    symptom_5 = request.form.get('Symptom_5')
    symptom_6 = request.form.get('Symptom_6')
    symptom_7 = request.form.get('Symptom_7')

    query = np.array([symptom_1, symptom_2, symptom_3, symptom_4, symptom_5, symptom_6, symptom_7])
    query = query.reshape(1, -1)
    predicted = model.predict(encoder.transform(query))[0]

    return jsonify({"Disease": predicted})

if __name__ == '__main__':
    app.run()