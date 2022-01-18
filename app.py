from flask import Flask, render_template, request, jsonify
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

@app.route("/Insinput", methods=["POST", "GET"])
def Insinput():
    return render_template("Insinput.html")

@app.route("/Inssubmit", methods=["POST", "GET"])
def Inssubmit():
    if request.method == "POST": 
        months_as_customer = float(request.form['months_as_customer'])
        age = float(request.form['age'])
        policy_number = float(request.form['policy_number'])
        policy_deductable = float(request.form['policy_deductable'])
        policy_annual_premium = float(request.form['policy_annual_premium'])
        umbrella_limit = float(request.form['umbrella_limit'])
        insured_zip = float(request.form['insured_zip'])
        capital_gains = float(request.form['capital-gains'])
        capital_loss = float(request.form['capital-loss'])
        incident_hour_of_the_day = float(request.form['incident_hour_of_the_day'])
        number_of_vehicles_involved = float(request.form['number_of_vehicles_involved'])
        bodily_injuries = float(request.form['bodily_injuries'])
        witnesses = float(request.form['witnesses'])
        total_claim_amount = float(request.form['total_claim_amount'])
        injury_claim = float(request.form['injury_claim'])
        property_claim = float(request.form['property_claim'])
        vehicle_claim = float(request.form['vehicle_claim'])
        auto_year = float(request.form['auto_year'])
        test = [[months_as_customer, age, policy_number, policy_deductable, policy_annual_premium, 
        umbrella_limit, insured_zip, capital_gains, capital_loss, incident_hour_of_the_day, 
        number_of_vehicles_involved, bodily_injuries, witnesses, total_claim_amount, injury_claim, 
        property_claim, vehicle_claim, auto_year]]
        model = pickle.load(open("Insurance_Fraud_Prediction/Insurance_Fraud_model.pkl","rb"))
        test = np.array(test)
        ans = model.predict(test)
        if ans[0] == 0:
            ans = "It is a Fraud"
        else:
            ans = "Not a Fraud"
        return render_template("Insprediction.html", result=ans)    


if __name__ == "__main__":
    app.run()