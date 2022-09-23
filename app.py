from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        model_name = request.form.get("models")
        model = joblib.load("pickle/"+model_name+".pkl")
        
        # Get values through input bars
        preg = request.form.get("pregnancies")
        glucose = request.form.get("glucose")
        bp = request.form.get("bp")
        skin = request.form.get("skinthic")
        insulin = request.form.get("insulin")
        bmi = request.form.get("bmi")
        dpf = request.form.get("dpf")
        age = request.form.get("age")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[preg,glucose,bp,skin,insulin,bmi,dpf,age]], columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # Get prediction
        prediction = model.predict_proba(X)[0][1]
        pred = round(prediction)
        percentage = round(prediction*100,2)
        op = ''
        if pred == 1:
            op = 'You have chances of having Diabetes'
        else:
            op = 'You don\'t have chances of having Diabetes'
        percent_str = 'Probability: '+str(percentage)+'%'
    else:
        op = ""
        percent_str = ''
        
    return render_template("website.html", output = op, percent = percent_str)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)