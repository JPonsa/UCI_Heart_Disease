# UCI - Heart Diseases

![heart ideases](./Figures/heart-disease-thumb.jpg)


## Summary

The goal of this project is to build a machine learning model that can predict whether a patient has heart disease based on various risk factors. The data used consists of medical records over 300 patients, including information on demographic factors (age, sex), symptoms (e.g.chest paint), and test results (e.g. resting electrocardiographic results). Each patient is labeled as having heart disease type 1-4 or not. 

To build the model, several classification algorithms were tested, including logistic regression, support vector machines, and random forests. The data was split into training and test sets, and cross-validation was used to evaluate model performance during training. The Support Vector Classifier achieved the highest accuracy on the test set of about 80%. These results are consistent with previous analysis. 

The most important features for predicting heart disease were found to be the presence of chest pain (type 2 and 3) and exercise induced angina. The trained model can now be used on new patient data to estimate the likelihood of heart disease, allowing for earlier intervention and improved outcomes as part of a **Clinical Decision Support Systems** (CDSS). Overall this project demonstrates the feasibility of using machine learning for disease prediction based on medical data.

## Goals:
- Produce a binary classification model that matches the performance reported in the Dataset Repo of Accuracy and Precisions over 80%.
- Produce a multi-lable classification model and compare the performance with the binary classification.
- Create a Clinical Decision Support System for of a UI interphase that allows users (e.g. clinicians) to enter an individual's parameters and obtain the probability of suffering from a heart disease

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

Please, see [Data > Original_dataset > heat-diease](./Data/Original_dataset/heart-disease.names) file provided by the authors for further detail on the dataset


## Technology
-
- 

## Observations

1. **Low correlation between the features.**

 A simple linear correlation (Pearson's r2) shows no relationship between a single feature and the targe(s) (num and num_01). The strongest correlation is observed with thalach, oldpeak, ca and thal. However, as we will se below, the last two features will be removed from the training dataset due to the large amount of missing values. 


![Feature Correlation](./Figures/10.feature_correlation.png)

2. **Large number of missing values.**

- VA Long Beach will be removed entirely from future analysis and the final training dataset.

- Features chol, fbs, ca, thal are also excluded from the any further analysis, including the training and assessment of ML models. In future iterations I will try to impute them in order to increase the training dataset. 

![Missing Values](./Figures/10.missing_values.png)

3. **SVC is the best Binary Classification Model.**

SVC is the winner of the model competition with an F1 score of 82% (and Accuracy and Precision over 80%). However, all models performed very similarly.

Overall, I obtained similar results are reported in the Data Repository (Accuracy and Precision over 80%).However, in my analysis SVC performed much better than reported in the Data Repository. 

![Binary Classfication CV](./Figures/40.binary_classifier_model_selection.png)

Note: The score for best model based on each metric is highlighted in red.

4. **Chest pain and exercise induced angina as the most important predictive features**

SHAP values were computed as measure of feature importance. Shapley values are a method from cooperative game theory applied to machine learning. They assign a value to each feature, quantifying its contribution to a model's prediction. Shapley values help explain the "credit" each feature receives in the prediction, aiding interpretability and fairness analysis. For more information about SHAP values please visit [shap.readthedocs.io > Introduction](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)

<img src="./Figures/40.binary_classifier_shap_values.png" alt="drawing" width="600"/>

## Lessons Learned
- 
## References:
 [1] Janosi,Andras, Steinbrunn,William, Pfisterer,Matthias, and Detrano,Robert. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.