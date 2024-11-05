from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('model/crop_model.pkl', 'rb') as f:
    rfc, ms, sc, crop_dict = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled_features = sc.transform(ms.transform(features))
        prediction = rfc.predict(scaled_features).reshape(1, -1)[0][0]

        crop_name = {v: k for k, v in crop_dict.items()}[prediction]
        return render_template('index.html', prediction_text=f"The best crop to be cultivated is {crop_name}.")
    except:
        return render_template('index.html', prediction_text="Error in input. Please check your values and try again.")

if __name__ == "__main__":
    app.run(debug=True)
