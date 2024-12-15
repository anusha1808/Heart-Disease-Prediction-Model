from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('heart_disease_predictor.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/heart-disease-info')
def heart_disease_info():
    return render_template('info.html')

@app.route('/heart-disease-predict', methods=['POST'])
def predict_result():
    try:
        # Collect input data from the form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Prepare the input features
        input_features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

        # Make prediction
        prediction = model.predict(input_features)
        result = "High Risk" if prediction[0] == 1 else "Low Risk"

        return render_template('result.html', result=result)
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
  app.run(debug=True, port=5000)