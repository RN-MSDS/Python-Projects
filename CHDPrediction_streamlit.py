#!/usr/bin/env python
# coding: utf-8

# # Coronary Heart Disease (CHD) Prediction with Logistic Regression
# # Machine Learning Project
# ## Performed by Ryan Navarro, BSN, RN, CPAN, CCRN Alumnus

# In[ ]:





# ### Source:
# [https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)
# 
# The dataset is from a CHD study from individuals living in Framingham, Massachusetts. 
# 
# The study identifies risk factors that contribute to CHD, such as hypertension, cholesterol levels, diabetes, blood glucose levels, etc. 
# 
# The goal is to predict whether an individual is at risk of developing CHD within the next 10 years. 
# 
# For the column 'TenYearCHD':
# - 1 means the individual is predicted to develop CHD within 10 years
# - 0 means the individual is not predicted to develop CHD within 10 years

# In[ ]:





# ### Get the Data

# In[4]:


import pandas as pd
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
'exec(% matplotlib inline)'
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns


# In[5]:


CHD = pd.read_csv('framingham.csv')


# ```
# ---
# Loads the dataset.
# ---
# ```

# In[7]:


CHD.shape


# ```
# ---
# Dataset contains:
# 
# 4240 Rows
# 16 Columns
# ---
# ```

# In[9]:


CHD.info()


# ```
# ---
# Null values noted. Column names shown. All columns have a numerical data type.
# 
# 'TenYearCHD' (TARGET) with no null values noted.
# ---
# ```

# In[11]:


CHD['TenYearCHD'].value_counts()


# ```
# ---
# Total value counts for the column 'TenYearCHD'.
# ---
# ```

# In[13]:


CHD = CHD.fillna(CHD.median())


# ```
# ---
# Null values imputed with median value (less sensitive to outliers) for each particular column.
# ---
# ```

# In[15]:


CHD.isnull().sum()


# ```
# ---
# No null values for each column after imputation with median value.
# ---
# ```

# In[17]:


CHD.head()


# ```
# ---
# First 5 rows of the dataset.
# ---
# ```

# In[19]:


CHD.describe()


# ```
# ---
# Descriptive statistics of all numerical columns of the dataset.
# ---
# ```

# In[21]:


CHD_target = CHD['TenYearCHD']

CHD_target


# ```
# ---
# Column 'TenYearCHD' -> TARGET/LABEL
# ---
# ```

# In[23]:


CHD_features = CHD.drop(columns=['TenYearCHD'])


# ```
# ---
# All the other columns excluding `TenYearCHD` -> FEATURES/ATTRIBUTES (15 total)
# ---
# ```

# In[25]:


CHD_features.info()


# In[26]:


scaler = preprocessing.StandardScaler()

CHD_features = scaler.fit(CHD_features).transform(CHD_features)

CHD_features


# ```
# ---
# 'CHD_features' standardization with use of StandardScaler() and then use of .fit() and .transform().
# ---
# ```

# In[28]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(CHD_features, CHD_target, test_size=0.20, random_state=42)

print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)


# ```
# ---
# Train-Test-Split (Split the dataset into 80% Train and 20% Test) and then shuffle using random_state.
# ---
# ```

# In[ ]:





# ### Explore the Data

# In[31]:


num_cols = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 
            'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD']

num_cols


# ```
# ---
# 'num_cols' -> all numerical columns of the dataset
# ---
# ```

# In[33]:


corr_matrix = CHD[num_cols].corr()

corr_matrix


# ```
# ---
# CORRELATION MATRIX of numerical columns.
# 
# 1 = Perfect Positive Correlation
# 0.7 to 1 = Strong Positive Correlation
# 0.4 to 0.7 = Moderate Positive Correlation
# 0 to 0.4 = Weak Positive Correlation
# 
# 0 = No Correlation
# 
# -0.4 to 0 = Weak Negative Correlation
# -0.7 to -0.4 = Moderate Negative Correlation
# -1 to -0.7 = Strong Negative Correlation
# -1 = Perfect Negative Correlation
# ---
# ```

# In[35]:


corr_matrix['TenYearCHD'].sort_values(ascending=False)


# ```
# ---
# List (from highest to lowest) of Features/Attributes with highest correlation to Target/Label ('TenYearCHD').
# ---
# ```

# In[37]:


plt.figure(figsize=(15,5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidth=1)
plt.title('Correlation Heatmap', fontsize=15)
plt.show()


# In[38]:


plt.rc('font', size=8)
plt.rc('axes', labelsize=8, titlesize=20)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

CHD.hist(bins=25, figsize=(25,25))
plt.suptitle('Histogram', fontsize=30)
plt.show()


# In[39]:


from pandas.plotting import scatter_matrix

scatter_matrix(CHD[num_cols], figsize=(30,30), s=25)
plt.suptitle('Scatterplot', fontsize=35)
plt.show()


# In[ ]:





# ### Prepare the Data

# In[41]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg


# ```
# ---
# Instantiate class of Logistic Regression.
# ---
# ```

# In[43]:


log_reg.fit(X_train, y_train)


# ```
# ---
# .fit() training data with Logistic Regression model.
# ---
# ```

# In[45]:


y_pred = log_reg.predict(X_test)

y_pred


# ```
# ---
# Prediction using trained Logistic Regression model 'log_reg' on the test data 'X_test' and store predicted class labels in 'y_pred'.
# ---
# ```

# In[ ]:





# ### Measure Performance on Test Set

# In[48]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

print('Accuracy Score of the Linear Regression model is:', accuracy_score(y_test, y_pred))


# In[49]:


from sklearn.model_selection import cross_val_score

cvs_log_reg = -cross_val_score(log_reg, X_train, y_train, scoring='neg_root_mean_squared_error', cv=3)

initial_rmse_log_reg = np.mean(cvs_log_reg)

print('Initial RMSE of the Linear Regression model is:', initial_rmse_log_reg)


# In[50]:


from sklearn.metrics import mean_squared_error

final_rmse_log_reg = mean_squared_error(y_test, y_pred, squared=False)

print('Final RMSE of the Linear Regression model is:', final_rmse_log_reg)


# In[51]:


from sklearn.metrics import r2_score

r2_score_log_reg = log_reg.score(X_test, y_test)

print ('R-squared of the Linear Regression model is:', r2_score_log_reg)


# In[ ]:





# ### Using Logistic Regression Model on New Data

# In[53]:


import pickle

with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(log_reg, file)

print("Model saved as 'logistic_regression_model.pkl'")


# ```
# ---
# Saves the trained model.
# ---
# ```

# In[55]:


with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("StandardScaler() saves as 'scaler.pkl'")


# In[ ]:





# ### Streamlit User Interface

# In[57]:


# Import libraries

import numpy as np
import pickle
import streamlit as st


# In[58]:


# Load files

with open('logistic_regression_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)


# ```
# ---
# 2 pickle files:
# 
# logistic_regression_model.pkl -> contains the trained model
# scaler.pkl -> contains the fitted scalers (or any other preprocessing transformer)
# ---
# ```

# In[60]:


# Create prediction function

def CHD_prediction(input_data):
    input_data_nparray = np.asarray(input_data) # Change data into NumPy array
    input_data_reshaped = input_data_nparray.reshape(1, -1) # Reshape array with 1 column (1) and as many rows to accomodate data (-1)
    
    new_data_scaled = loaded_scaler.transform(input_data_reshaped) 
    prediction = loaded_model.predict(new_data_scaled)

    if prediction == 0:
        return 'Individual is NOT PREDICTED to develop Coronary Heart Disease (CHD) within the next 10 years'
    else:
        return 'Individual IS PREDICTED to develop Coronary Heart Disease (CHD) within the next 10 years'

def main():
    # Title
    st.title('10-year risk prediction for developing CHD')

    # User input
    male = st.text_input('Gender (1 = male, 2 = female):')
    age = st.text_input('Age:')
    education = st.text_input('Education (1 = some HS, 2 = HS grad, 3 = some college, 4 = college grad):')
    currentSmoker = st.text_input('Curent smoker (1 = yes, 2 = no):')
    cigsPerDay = st.text_input('Number of cigarettes per day:')
    BPMeds = st.text_input('Takes blood pressure medication(s) (1 = yes, 2 = no):')
    prevalentStroke = st.text_input('History of stroke (1 = yes, 2 = no):')
    prevalentHyp = st.text_input('History of high blood pressure (1 = yes, 2 = no):')
    diabetes = st.text_input('History of diabetes (1 = yes, 2 = no):')
    totChol = st.text_input('Total cholesterol:')
    sysBP = st.text_input('Systolic blood pressure:')
    diaBP = st.text_input('Diastolic blood pressure:')
    BMI = st.text_input('BMI:')
    heartRate = st.text_input('Heart rate:')
    glucose = st.text_input('Blood sugar:')

    # CHD prediction
    dx = ''

    # Button for CHD prediction
    if st.button('CHD Prediction'):
        dx = CHD_prediction(
            [male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP,
            diaBP, BMI, heartRate, glucose])
    st.success(dx)

if __name__ == '__main__':
    main()

