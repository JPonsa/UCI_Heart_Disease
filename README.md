# UCI - Heart Diseases

![heart ideases](https://github.com/JPonsa/UCI_Heart_Disease/blob/main/Figures/heart-disease-thumb.jpg)


## Summary

The goal of this project is to build a machine learning model that can predict whether a patient has heart disease based on various risk factors. The data used consists of medical records over 300 patients, including information on demographic factors (age, sex), symptoms (e.g.chest paint), and test results (e.g. resting electrocardiographic results). Each patient is labeled as having heart disease type 1-4 or not. 

To build the model, several classification algorithms were tested, including logistic regression, support vector machines, and random forests. The data was split into training and test sets, and cross-validation was used to evaluate model performance during training. The Support Vector Classifier achieved the highest accuracy on the test set of about 80% accuracy. These results are consistent with previous analysis. 

The most important features for predicting heart disease were found to be the presence of chest pain (type 2 and 3) and exercise induced angina. The trained model can now be used on new patient data to estimate the likelihood of heart disease, allowing for earlier intervention and improved outcomes as part of a **Clinical Decision Support Systems** (CDSS). Overall this project demonstrates the feasibility of using machine learning for disease prediction based on medical data.

## Goals:
- Produce a binary classification model that matches the performance reported in the Dataset Repo of Accuracy and Precisions over 80%. <img src="https://upload.wikimedia.org/wikipedia/commons/8/8c/White_check_mark_in_dark_green_rounded_square.svg" widht="15" height="15"/>

- Produce a multi-lable classification model and compare the performance with the binary classification. <img src="https://upload.wikimedia.org/wikipedia/commons/c/cc/Cross_red_circle.svg" widht="15" height="15"/>

- Create a Clinical Decision Support System for of a UI interphase that allows users (e.g. clinicians) to enter an individual's parameters and obtain the probability of suffering from a heart disease. <img src="https://upload.wikimedia.org/wikipedia/commons/8/8c/White_check_mark_in_dark_green_rounded_square.svg" widht="15" height="15"/>

## About the data
The UCI Heart Diseases dataset is the combination of 4 databases (Cleveland, Hungary, Switzerland, and the VA Long Beach)

<details>

    Features:
    - age: age in years
    - sex: sex (1 = male; 0 = female)
    - cp: chest pain type

        1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 =asymptomatic

    - trestbps: resting blood pressure (in mm Hg on admission to the 
            hospital)
    - chol: serum cholestoral in mg/dl
    - restecg: resting electrocardiographic results

        0 = normal.

        1 = having ST-T wave abnormality (T wave inversions and/or ST 
            elevation or depression of > 0.05 mV).

        2 = showing probable or definite left ventricular hypertrophy
            by Estes' criteria.

    - thalach: maximum heart rate achieved
    - exang: exercise induced angina (1 = yes; 0 = no)
    - slope: the slope of the peak exercise ST segment

        1 = upsloping; 2 = flat; 3 = downsloping

    - ca: number of major vessels (0-3) colored by flourosopy
    - thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

    Target(s)
    - num: heart disease (0 =  healthy, 1-4 = heart disease) 
    - num_01 : binary version of "num" (0 =  healthy, 1= heart disease) 

</details>

.

Data Source: https://archive.ics.uci.edu/dataset/45/heart+disease

Please, see [Data > Original_dataset > heat-diease](./data/Original_dataset/heart-disease.names) file provided by the authors for further detail on the dataset

## Observations

1. **Low correlation between the features.**

 A simple linear correlation (Pearson's r2) shows no strong relationship between a single feature and the targe(s) (num and num_01). The strongest correlation is observed with thalach, oldpeak, ca and thal. However, as we will se below, the last two features will be removed from the training dataset due to the large amount of missing values. 


![Feature Correlation](https://github.com/JPonsa/UCI_Heart_Disease/blob/main/Figures/10.feature_correlation.png)

2. **Large number of missing values.**

- VA Long Beach will be removed entirely from future analysis and the final training dataset.

- Features chol, fbs, ca, thal are also excluded from the any further analysis, including the training and assessment of ML models. In future iterations I will try to impute them in order to increase the training dataset. 

![Missing Values](https://github.com/JPonsa/UCI_Heart_Disease/blob/main/Figures/10.missing_values.png)

3. **SVC is the best Binary Classification Model.**

SVC is the winner of the model competition with an F1 score of 82% (and Accuracy and Precision over 80%). However, all models performed very similarly.

Overall, I obtained similar results are reported in the Data Repository (Accuracy and Precision over 80%).However, in my analysis SVC performed much better than reported in the Data Repository. 

![Binary Classification CV](https://github.com/JPonsa/UCI_Heart_Disease/blob/main/Figures/40.binary_classifier_model_selection.png)

Note: The score for best model based on each metric is highlighted in red.

Confusion matrix for the SVM after hyper-parameter tunning.

![Best Binary Classification CM](https://github.com/JPonsa/UCI_Heart_Disease/blob/main/Figures/40.binary_confusion_matrix.png)

4. **Chest pain and exercise induced angina as the most important predictive features**

SHAP values were computed as measure of feature importance. Shapley values are a method from cooperative game theory applied to machine learning. They assign a value to each feature, quantifying its contribution to a model's prediction. Shapley values help explain the "credit" each feature receives in the prediction, aiding interpretability and fairness analysis. For more information about SHAP values please visit [shap.readthedocs.io > Introduction](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)

<img src="https://archive.ics.uci.edu/dataset/45/heart+disease/42.binary_classifier_shap_values.png" alt="drawing" width="600"/>

## CDSS with Streamlit.

A simple web application was created using Streamlit surfacing the trained model to
possible end-users. This application allows them to enter a set of medical metrics and understand the likelihood of suffering from a heart condition. 

### Home page
The home page contains the following sections:
- **About** - A short description of the Project and the Clinical Decission Support System app.
- **How to use** - A clear description of the steps that the user must follow in order to input the patient's metrics. It also includes a description of the analysis performed and how to interpret it.
- **Data** - A short description of the key parameters used to predict heart diseases and the source of the training data.

<img src="https://archive.ics.uci.edu/dataset/45/heart+diseasecdss_streamlit_patient_home.png" alt="drawing" width="600"/>

### Patient Assessment
This is the core of the application and contains:
- **Patient settings** - a set of objects that enable the user the easily introduce of patient metrics. The parameters are already set to the most frequent value observed based on the training dataset to facilitate the task.
- **Patient Assessment** - The result of the prediction, indicating the category (Healty or Heart Diseases) and the probability (from 0 to 1) associated with the prediction. It also includes the analysis of the influence of the different metrics on the prediction.

<img src="https://archive.ics.uci.edu/dataset/45/heart+disease/cdss_streamlit_patient_assessment.png" alt="drawing" width="600"/>

## Future work
- Produce a multi-lable classification model and compare the performance with the binary classification. I attempted it but the performance was very poor.No model and configuration I tried was able to differentiate between the different types of heart disease. In future iterations I will try to implement a Synthetic Minority Oversampling Technique (SMOTE)[3]

- Impute missing values. This could help with the performance of the multi-label classifier by increasing the training dataset.

- Improve the CDSS UI.

- Deploy app into Streamlit Community Cloud so it can be used by others.

## References:
 [1] Janosi,Andras, Steinbrunn,William, Pfisterer,Matthias, and Detrano,Robert. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.

 [2] NHS Streamlit App template. https://github.com/nhs-pycom/nhs-streamlit-template

 [3] Chawla, N.V. et al., 2002. SMOTE: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16, pp.321–357.