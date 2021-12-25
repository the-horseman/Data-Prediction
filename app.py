from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route("/")
def fist_page():
    return render_template("choice.html")

@app.route("/Wineinput", methods=["POST", "GET"])
def Wineinput():
    return render_template("Wineinput.html")

@app.route("/Winesubmit", methods=["POST", "GET"])
def Winesubmit():
    if request.method == "POST":
        fixed_acidity = float(request.form['fixed acidity'])
        volatile_acidity = float(request.form['volatile acidity'])
        citric_acid = float(request.form['citric acid'])
        residual_sugar = float(request.form['residual sugar'])
        chlorides = float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['free sulfur dioxide'])
        total_sulfur_dioxide = float(request.form['total sulfur dioxide'])
        density = float(request.form['density'])
        pH = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])
        test = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
        total_sulfur_dioxide, density, pH, sulphates, alcohol]]
        model = pickle.load(open("Wine_Quality_Prediction/Wine_Quality_model.pkl","rb"))
        test = np.array(test)
        ans = model.predict(test)
        return render_template("Wineprediction.html", result=ans[0])    



if __name__ == "__main__":
    app.run(debug=True)