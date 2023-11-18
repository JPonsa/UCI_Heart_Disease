# Heart Disease Clinical Decision Support System (CDSS)
# Author: Joan Ponsa

import streamlit as st
import pandas as pd
from dataclasses import dataclass

import joblib
import pickle
import shap
import matplotlib.pyplot as plt


@dataclass
class Patient:
    age: float = 50
    sex: float = 0
    cp_1: int = 0
    cp_2: int = 0
    cp_3: int = 0
    trestbps: float = 150
    restecg_1: int = 0
    restecg_2: int = 0
    thalach: float = 150
    exang: float = 0
    oldpeak: float = 0
    slope: float = 0

    def items(self):
        return {
            "age": [self.age],
            "sex": [self.sex],
            "cp_1": [self.cp_1],
            "cp_2": [self.cp_2],
            "cp_3": [self.cp_3],
            "trestbps": [self.trestbps],
            "restecg_1": [self.restecg_1],
            "restecg_2": [self.restecg_2],
            "thalach": [self.thalach],
            "exang": [self.exang],
            "oldpeak": [self.oldpeak],
            "slope": [self.slope],
        }


# Load model and exapliner
# @st.cache_resource
def load_binary_classifier():
    model_filename = "./models/best_binary_classifier.pkl"
    with open(model_filename, "rb") as f:
        pipeline = joblib.load(f)

    with open("./models/best_binary_classifier_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    return pipeline, explainer


# write home page
def home_page():
    # refactor using markdown
    st.markdown(
        """
        ### About the Project
        This project presents a Clinical Decision Support System (CDSS) that leverages machine learning to predict the likelihood of heart disease in patients. The system is trained on a variety of health metrics to make its predictions.
        A CDSS is a tool designed to aid healthcare professionals in making decisions about patient care. This particular CDSS provides an estimation of a patient's risk of heart disease, and outlines the factors contributing to this prediction.

        **Disclaimer: This CDSS is a tool for support and should not be used as a sole resource for medical diagnosis. It is not an official NHS application and its use should be combined with professional medical advice.**
        
        For more information about the procject, please visit: 
        """
    )

    how_to_use = st.expander("How to Use")
    with how_to_use:
        st.write(
            """
        To use the CDSS, enter the patient's health metrics in the form on the left. 
        Then click the 'Run Assessment' button. 
        The system will predict whether the patient is likely to have heart disease, 
        and display the influence of each metric in the prediction.
        """
        )

    data = st.expander("Data")
    with data:
        st.write(
            """
            The UCI Heart Diseases dataset is the combination of 4 databases (Cleveland, Hungary, Switzerland, and the VA Long Beach)
            
            Features:
            
            - age: age in years.
            
            - sex: gender (1 = male; 0 = female).
            
            - cp: chest pain type
            
                0 = asymptomatic
            
                1 = typical angina 
            
                2 = atypical angina 
            
                3 = non-anginal pain; 

            - trestbps: resting blood pressure (in mm Hg on admission to the hospital)
            
            - chol: serum cholestoral in mg/dl
            
            - restecg: resting electrocardiographic results
               
                0 = normal.
               
                1 = having ST-T wave abnormality (T wave inversions and/or ST 
                    elevation or depression of > 0.05 mV).
               
                2 = showing probable or definite left ventricular hypertrophy
                    by Estes' criteria.

            
            - thalach: maximum heart rate achieved
            
            - exang: exercise induced angina (1 = yes; 0 = no)
            
            - oldpeak = ST depression induced by exercise relative to rest
            
            - slope: the slope of the peak exercise ST segment
                
                1 = upsloping; 
                
                0 = flat; 
                
                -1 = downsloping.
                    
            For more information about the dataset, please visit: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
        """
        )


def run_assessment(patient: Patient):
    cdss_data = pd.DataFrame(patient.items())
    cdss_data = cdss_data[features]  # reoder features
    shap_values = explainer(preporcessor.transform(cdss_data))

    y_pred = pipeline.predict(cdss_data)
    y_pred_proba = pipeline.predict_proba(cdss_data)

    st.write(f"Prediction : {'Healthy'if y_pred[0]==0 else 'Heart Disease'}")
    st.write(f"Probability : {y_pred_proba[0][y_pred[0]]:.2f}")

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)

    force_fig, ax = plt.subplots()
    force_plot = shap.force_plot(
        explainer.expected_value, shap_values[0].values, cdss_data, matplotlib=True
    )
    st.pyplot(force_plot, clear_figure=True)

    waterfall_fig, ax = plt.subplots(figsize=(3, 6))
    ax = shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(waterfall_fig, clear_figure=True)

    st.write(
        """
        Note: Notice that the numbers displayed in the waterfall chart 
        are difference than those imputed. This is because the model is trained
        on rescaled data.
        """
    )


def heart_disease_assessment():
    c1, c2 = st.columns((1, 2))

    with c1:
        st.markdown(
            "<h2 style='text-decoration: underline; color: black; text-underline-position: under; text-decoration-color: blue;'>Patient Settings</h2>",
            unsafe_allow_html=True,
        )

        patient = Patient()

        patient.age = st.slider("Age", min_value=0, max_value=90, value=50, step=1)

        patient.sex = float(st.selectbox("Sex", ["Male", "Female"]) == "Male")

        chest_pain_types = [
            "0 - Asymptomatic",
            "1 - Typical angina",
            "2 - Atypical angina",
            "3 -  non-anginal pain",
        ]
        chest_pain = st.selectbox("Chest pain (coded as : **cp**)", chest_pain_types)
        patient.cp_1 = int(chest_pain == chest_pain_types[1])
        patient.cp_2 = int(chest_pain == chest_pain_types[2])
        patient.cp_3 = int(chest_pain == chest_pain_types[3])

        patient.trestbps = st.slider(
            "Resting blood pressure (mm Hg) (coded as :**trestbps**)", 75, 200, 150, 5
        )

        restecg_type = [
            "0 - Normal",
            "1 - ST-T wave abnormality",
            "2 - Left ventricular hypertrophy",
        ]
        restecg = st.selectbox(
            "Resting Electrocardiographic Results (coded as :**restecg**)", restecg_type
        )
        patient.restecg_1 = int(restecg == restecg_type[1])
        patient.restecg_1 = int(restecg == restecg_type[2])

        patient.thalach = st.slider(
            "Maximum Heart Rate achieved (coded as :**thalach**)", 50, 200, 150, 5
        )

        patient.exang = float(
            st.selectbox("Exercise Induced Angina (coded as: **exang**)", ["No", "Yes"])
            == "Yes"
        )

        patient.oldpeak = st.slider(
            "ST depression induced by exercise relative to rest (coded as :**oldpeak**)",
            0.0,
            7.0,
            0.0,
            step=0.5,
        )

        patient.slope = st.slider(
            "The **slope** of the peak exercise ST segment", -1, 1, 0, 1
        )

    with c2:
        # header with underline
        st.markdown(
            "<h2 style='text-decoration: underline; color: black; text-underline-position: under; text-decoration-color: blue;'>Patient Assessment</h2>",
            unsafe_allow_html=True,
        )

        button = st.button("Run Assessment")
        if not button:
            st.write("This may take a few seconds... Sorry for the incovenience")
        if button:
            run_assessment(patient)


# Binnary classifier features
# fmt:off
numeric_features = ["age", "trestbps", "thalach", "oldpeak", "slope"]
binary_features = ["sex", "cp_1", "cp_2", "cp_3", "restecg_1", "restecg_2", "exang"]
features = numeric_features + binary_features
# fmt:on


if __name__ == "__main__":
    shap.initjs()

    pipeline, explainer = load_binary_classifier()
    classifier = pipeline["classifier"]
    preporcessor = pipeline["preprocessor"]

    st.image("./figures/heart-ecg-black.jpg", width=400)
    st.title("Heart Disease Clinical Decision Support System")
    st.sidebar.title("Navigation")

    page = st.sidebar.radio("Pages", options=["Home", "Heart Disease Assessment"])

    if page == "Home":
        home_page()

    if page == "Heart Disease Assessment":
        heart_disease_assessment()
