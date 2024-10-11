from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

with open('C:/Users/ANIRUDH TV/Desktop/Lung_Cancer_Prediction_Using_Machine_Learning-main/model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = [
                float(request.form['gender']),
                float(request.form['age']),
                float(request.form['smoking']),
                float(request.form['yellow_fingers']),
                float(request.form['anxiety']),
                float(request.form['peer_pressure']),
                float(request.form['chronic_disease']),
                float(request.form['fatigue']),
                float(request.form['allergy']),
                float(request.form['wheezing']),
                float(request.form['alcohol_consuming']),
                float(request.form['coughing']),
                float(request.form['shortness_of_breath']),
                float(request.form['swallowing_difficulty']),
                float(request.form['chest_pain'])
            ]

            # Convert data to DataFrame for prediction
            df = pd.DataFrame([data])

            # Predict using the loaded model
            prediction = model.predict(df)

            # Render result template with prediction
            return render_template('result.html', prediction=prediction[0])
        except Exception as e:
            return str(e)

if __name__ == '__main__':
    app.run(debug=True)
