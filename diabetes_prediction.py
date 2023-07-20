#!/usr/bin/env python
# coding: utf-8

# ## Models exploring differnt machine learning techniques to predict the accuracy of getting diabetes

# The dataset is from here: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset?resource=download

# # Introduction

# The Diabetes prediction dataset is a collection of medical and demographic data from patients, along with their diabetes status (positive or negative). The data includes features such as age, gender, body mass index (BMI), hypertension, heart disease,smoking history, HbA1c level, and blood glucose level. This dataset can be used to build machine learning models to predict diabetes in patients based on their medical history and demographic information. This can be useful for healthcare professionals in identifying patients who may be at risk of developing diabetes and in developing personalized treatment plans. Additionally, the dataset can be used by researchers to explore the relationships between various medical and demographic factors and the likelihood of developing diabetes. The machine learning models that have been used to predict the outcome of getting diabetes are RandomForrestClassifier, DecisionTreeClassifier and XGB Boost.

# Age is an important factor as diabetes is more commonly diagnosed in older adults. Age ranges from 0-80 in our dataset.
# 
# Hypertension is a medical condition in which the blood pressure in the arteries is persistently elevated. It has values a 0 or 1 where 0 indicates they don’t have hypertension and for 1 it means they have hypertension.
# 
# Heart disease is another medical condition that is associated with an increased risk of developing diabetes. It has values a 0 or 1 where 0 indicates they don’t have heart disease and for 1 it means they have heart disease.
# 
# Smoking history is also considered a risk factor for diabetes and can exacerbate the complications associated with diabetes.In our dataset we have 6 categories i.e not current,former,No Info,current,never and ever.
# 
# 1)not current: individuals who used to smoke but are currently not smoking
# 2)former: individuals who used to smoke but have been abstinent for a long period of time.
# 3)No Info: smoking history is unknown
# 4)current: individuals who currently smoke
# 5)never: individuals who have never smoked
# 6)ever: individuals who have at one point been a smoker.
# 
# BMI (Body Mass Index) is a measure of body fat based on weight and height. Higher BMI values are linked to a higher risk of diabetes. The range of BMI in the dataset is from 10.16 to 71.55.
# 
# Less than 18.5 is underweight
# 18.5-24.9 is normal
# 25-29.9 is overweight
# 30 or more is obese.
# 
# HbA1c (Hemoglobin A1c) level is a measure of a person's average blood sugar level over the past 2-3 months. Higher levels indicate a greater risk of developing diabetes. Mostly more than 6.5% of HbA1c Level indicates diabetes.
# 
# Blood glucose level refers to the amount of glucose in the bloodstream at a given time. High blood glucose levels are a key indicator of diabetes.
# 
# The response variable diabetes holds binary values to determine whether an individual has diabetes (1) or not (0).

# In[214]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[215]:


import os


# In[216]:


os.chdir(r"C:\Users\sarit\Documents\Careerera\Own Practise")


# In[217]:


df=pd.read_csv("diabetes_prediction_dataset.csv")


# In[218]:


df.shape


# In[219]:


#Univariate Analysis


# In[220]:


df.dtypes


# In[221]:


df["smoking_history"].value_counts()


# In[222]:


df.isnull().sum()


# In[223]:


df["gender"].value_counts()


# In[224]:


df_dist=df["diabetes"].value_counts()/len(df)
df_dist


# In[225]:


plt.pie(df_dist, labels = ['non-diabetic', 'diabetic'],
       autopct = '%1.1f%%')
plt.title("Distribution of diabetics in dataset")
plt.show()


# This is a pie chart showing that around 90% of the dataset doesn't have diabetes. The purpose of this study is to predict if they have diabetes or not

# In[226]:


sns.catplot(data=df, kind = "count", x="smoking_history")
plt.title("Smoking History")
plt.show()


# This graph shows that a lot of people have decided not to disclose their smoking status or have never smoked in their life

# In[227]:


plt.hist(df["bmi"], bins=[0,5,10,15,20,25,30,35,40])
plt.title("BMI Range")
plt.show()


# From this you can tell majority people BMi are between 25 and 30 so are nearing obesity levels. The data isn't nomrally distributed

# In[228]:


sns.distplot(df["age"])
plt.xlabel("Age")
plt.ylabel("Density")
plt.title("Distrubtion of Age")
plt.show()


# Age isn't normally distributed

# In[229]:


df["blood_glucose_level"].value_counts()


# In[230]:


plt.hist(df["blood_glucose_level"], bins=[80,100,120,140,160,180,200,220,240,260,280,300])
plt.title("Blood Glucose Levles")
plt.show()


# Here you can see a histogram showing that majority people's blood sugar level is around 150. This is quite high and can lead to health issues such as diabetes. There is a small minority with over 200 mg/DL blood sugar levels which is extremely high

# In[231]:


df.columns


# In[232]:


df.describe(percentiles=[0.01,0.25,0.5,0.75,0.99]).T


# In[233]:


sns.countplot(x="heart_disease", data=df)
plt.show()


# Majority don't have a heart disease

# In[234]:


df.columns


# In[235]:


pd.DataFrame(df["HbA1c_level"].value_counts())


# In[236]:


sns.countplot(x="HbA1c_level", data=df)
plt.title("Average Hemoglobin A1c level over 2-3 months")
plt.show()


# As you can see above that majority have a hemoglobin level between 5.7 and 6.6 so don't have type 2 diabetes but are close to border line diabetes. For type 1 diabetes there is a high risk for them to be diabetic.

# In[237]:


sns.countplot(x="hypertension", data=df)
plt.title("Graph to show if blood arteties are constantly elevated")
plt.show()


# As you can see majority of the population aren't at risk of their blood pressure in their arteries being elevated

# In[238]:


def smoke(smoking_status):
    if smoking_status in ['never','No Info']:
        return 'non-smoker'
    elif smoking_status == 'current':
        return 'current'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'past-smoker'

df['smoking_history']=df['smoking_history'].apply(smoke)
df['smoking_history'].unique()


# I decided to clear up the smoking cateogories to make it clearer as we saw previously when conducting EDA it wasn't clear

# In[239]:


df["smoking_history"].value_counts()


# In[240]:


dummies=pd.get_dummies(df,columns=["gender", "smoking_history", ])
df1=pd.concat([df,dummies],axis=1)
df1


# In[241]:


df1.drop(columns=["gender_Other"], inplace=True)


# Decided to drop other gender cateogory as there is only 18 in the dataset.

# In[242]:


df1=df1.loc[:,~df1.columns.duplicated()]


# In[243]:


df1.drop(columns=["smoking_history", "gender"], inplace=True)


# In[244]:


df1.head()


# In[ ]:





# In[245]:


def outliers_percentile(x):
    x=x.clip(upper=x.quantile(.99))
    x=x.clip(lower=x.quantile(.01))
    return x


# In[246]:


df1=df1.apply(outliers_percentile)


# In[247]:


df1.describe().T


# I got rid of any anomolies within the dataset to get rid of skewness. 
# 
# Max age: 80
# Max BMI : 48.8
# Avg: BMI: 27.3
# Max Blood Gluscose Level: 280
# Max Hemoglobin level: 9

# In[ ]:





# # Bivariate Analysis

# I am grouping various variables to see one discrete variable and a continous compare by using min, mean,median and max

# In[248]:


df1.groupby(["diabetes"]).agg({"bmi": ["mean","min","median", "max"]})


# This shows that people with a diabette has an average bmi of 31.7 and people without diabetes has an average bmi of 26 which is near the borderline of being obese. 

# In[249]:


df1.groupby(["blood_glucose_level"]).agg({"bmi": ["mean","min","median", "max"]})


# In[250]:


sns.boxplot(data=df1 , x=df1["diabetes"], y = df1["age"])


# Most diabetes is between the age 50 and 80

# In[251]:


sns.boxplot(data=df1 , x=df1["diabetes"], y = df1["bmi"])


# Most people that has diabetes has a bmi between 25 and 35

# In[252]:


#Correlation


# In[253]:


cr=df1.corr()
plt.figure(figsize=(10,10))
#cr1=cr[cr>=0.7]
#matrix = np.triu(cr1)
sns.heatmap(cr, annot=True, cmap="coolwarm")


# There isn't any real high correlation between variables except betweensmoking_history_current and smoking_history_past_smoker ans smoking_history_non_smoker and smoking_history_past_smoker. There is also a slightly higher correlation between age and bmi and age an diabetes as expected

# I am preparing my data to help predict the outcome. Diabetes is the y variable

# # Model Development

# In[254]:


df2=df1


# In[255]:


y=df2["diabetes"]
x=df2.drop(columns=["diabetes"])


# In[256]:


from sklearn.model_selection import train_test_split


# In[257]:


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,  random_state=0)


# As our target variable is diabetes but 90% of the dataset don't have diabetes, so the date is skewed to the right. Smote and Rus have to be applied where smote increases a % of the people with diabetes in the dataset and random sampler reduces the % of the people without diabetes within the dataset. 

# In[258]:


#pip install imbalanced-learn


# In[259]:


#pip install --upgrade scikit-learn imbalanced-learn


# In[260]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# In[261]:


smote = SMOTE(sampling_strategy=0.1) #set to 0.1, which means the minority class will be increased to 10% of the total dataset.
rus = RandomUnderSampler(sampling_strategy=0.5)#set to 0.5, which means the majority class will be reduced to 50% of the total dataset.


# In[262]:


x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
x_train_resampled, y_train_resampled = rus.fit_resample(x_train, y_train)


# In[263]:


x_train_resampled.size


# In[264]:


x_test_resampled, y_test_resampled = smote.fit_resample(x_train, y_train)
x_test_resampled, y_test_resampled = rus.fit_resample(x_train, y_train)


# In[265]:


x_test_resampled.size


# In[266]:


from sklearn.ensemble import RandomForestClassifier


# In[267]:


from sklearn.tree import DecisionTreeClassifier


# In[268]:


Rf = RandomForestClassifier()


# In[269]:


Rf.fit(x_train_resampled, y_train_resampled)


# In[270]:


# Evaluating a model on test data
Rf.score(x_test , y_test)


# In[271]:


Rf.score(x_train, y_train)


# In[272]:


Rf.score(x_test_resampled , y_test_resampled)


# In[273]:


Rf.score(x_train_resampled , y_train_resampled)


# For the orginal data there is nearly 100% accurate in predicting the outcome of having diabetes

# In[274]:


Dt = DecisionTreeClassifier()


# In[275]:


Dt.fit(x_train_resampled, y_train_resampled)


# In[276]:


# Evaluating a model on test data
Dt.score(x_test , y_test)


# In[277]:


Dt.score(x_train , y_train)


# Decision tree classifier is less accurate in predicting the outcome of having diabetes compared to the Random Forest. The Decision Tree Classiifer train and test result is fairly similar

# # Modelling

# In[278]:


pred_test=Rf.predict(x_test)
pred_train=Rf.predict(x_train)


# In[279]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[280]:


print(len(y_test))
print(len(pred_test))


# In[281]:


metrics.confusion_matrix(y_train,pred_train)


# In[283]:


print(metrics.classification_report(y_test,pred_test))


# In[284]:


print(metrics.classification_report(y_train,pred_train))


# In[285]:


pred_test_resampled=Rf.predict(x_test_resampled)
pred_train_resampled=Rf.predict(x_train_resampled)


# In[286]:


print(len(y_test_resampled))
print(len(pred_test_resampled))


# In[287]:


print(metrics.classification_report(y_test_resampled,pred_test_resampled))


# In[288]:


print(metrics.classification_report(y_train_resampled,pred_train_resampled))


# In[ ]:





# Precision is a measure of how many variables have been correctly predicted. We can see that 59% are True Positives and 98% are True Negatives. For the trained data, 69% are True Postivies and 100% are True Negative. In this instance 1 is the target variable of someone having diabetes. 
# 
# Recalll is the actual instances the classes have been predicted.The test data predicted 82% corrrectly whilst the trained data predicted all the people with diabetes corrrectly.
# 
# F1-Score is the harmonic mean of precsion and prvodies a balance between 2 metrics.
# 
# You can also how the orginal data is skewed. When i used the resampled data, the precision for predicting someone with diabetes for the test data increases to 92%. It has been underfitted which would be expected as 90% of the orginal data is skewed towards them not having diabetes

# # XGB Boost

# In[289]:


from xgboost import XGBClassifier


# In[290]:


from sklearn.metrics import accuracy_score


# In[291]:


xbg_model = XGBClassifier(n_estimators=1000, learning_rate=0.05,n_jobs=4,early_stopping_rounds=5)
xbg_model.fit(x_train, y_train, eval_set=[(x_test, y_test)],verbose=False)
predictions = xbg_model.predict(x_test)
print("Accuracy of model: ", accuracy_score(y_test, predictions))
print(metrics.classification_report(y_test, predictions))


# In[292]:


xbg_model = XGBClassifier(n_estimators=1000, learning_rate=0.05,n_jobs=4,early_stopping_rounds=5)
xbg_model.fit(x_train_resampled, y_train_resampled, eval_set=[(x_test_resampled, y_test_resampled)],verbose=False)
predictions = xbg_model.predict(x_test_resampled)
print("Accuracy of model: ", accuracy_score(y_test_resampled, predictions))
print(metrics.classification_report(y_test_resampled, predictions))


# Here you can see the resampled data has a 91% precision, the XGB model prediction is 94% acccurate whilst the normal data has been overfitted and has an accuracy of 97% with a 96% precision. 

# # Conclusion

# Overall XGB and Random Forest both give a fairly accurate prediction of someone having diabetes

# In[ ]:




