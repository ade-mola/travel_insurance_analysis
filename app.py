import pickle
import pandas as pd
import numpy as np
import streamlit as st


class TravelInsuranceModel:
    def __init__(self, model_file_path):
        self.model = self.load_model(model_file_path)

    def load_model(self, model_file_path):
        with open(model_file_path, "rb") as f:
            return pickle.load(f)

    def predict_views(self, data):
        x = pd.DataFrame(data, index=[0])

        # x.AnnualIncome = int(x.AnnualIncome)
        # bins = [ 300000, 800000, 1300000, 1800000]
        # bands = ["Band 1", "Band 2", "Band 3"]
        # x["IncomeBands"] = pd.cut(x.AnnualIncome, bins, labels=bands)

        # x["AgeGroup"] = np.where(int(x["Age"]) < 30, "20-29", "30-39")

        cat_cols = ['EmploymentType', 'GraduateOrNot', 'ChronicDiseases', 
                    'FrequentFlyer', 'EverTravelledAbroad']
        
        x[cat_cols] = x[cat_cols].apply(lambda x: x.astype("category"))
        for col in cat_cols:
            x[col] = x[col].cat.codes

        return int(self.model.predict(x)[0])


class TravelInsuranceApp:
    def __init__(self):
        self.model = TravelInsuranceModel("./models/travel_insurance_model.pickle")

    def show_header(self):
        st.title("Insure Your Travel: AI-Powered Prediction")
        st.markdown(
            """
        This web application is an implementation of a predictive model 
        for predicting if a customer would purchase travel insurance or not.
        The model takes inputs such as the customer's age, employment type, 
        income, and others, and predicts if they will buy a travel insurance or not.

        <a href="https://github.com/ade-mola/travel_insurance_analysis">GitHub Codebase</a>

        <div style="background-color:#f63366;
                    border-radius: 25px;
                    padding:5px">
            <h2 style="color:white;
                       text-align:center;">
                Travel Insurance Predictor: Smart App
            </h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def get_input_data(self):
        age = st.text_input("Enter Age: ")
        employment_type = st.radio(
            "Employment Type: ", ["Government Sector", "Private Sector/Self Employed"]
        )
        graduate = st.radio("Graduated College? : ", ["Yes", "No"])
        income = st.number_input('Input Annual Income:', 300000, 1800000)
        family_members = st.text_input("Size of Family Members: ")
        chronic_disease = st.radio("Any history of chronic disease? : ", ["Yes", "No"])
        frequent_flyer = st.radio("Are you a frequent flyer? : ", ["Yes", "No"])
        travelled_abroad = st.radio("Have you ever traveled abroad ? : ", ["Yes", "No"])
        

        data = {
            "Age": age,
            "EmploymentType": employment_type,
            "GraduateOrNot": graduate,
            "AnnualIncome": income,
            "FamilyMembers": family_members,
            "ChronicDiseases": chronic_disease,
            "FrequentFlyer": frequent_flyer,
            "EverTravelledAbroad": travelled_abroad,
        }

        return data

    def show_prediction_result(self, purchase):
        st.success(f"Likely to purchase travel insurance? : {purchase}")

    def run(self):
        self.show_header()
        data = self.get_input_data()

        if st.button("Predict"):
            purchase = self.model.predict_views(data)
            if purchase == 1:
                purchase = 'Yes'
            else:
                purchase = 'No'
            self.show_prediction_result(purchase)


if __name__ == "__main__":
    app = TravelInsuranceApp()
    app.run()
