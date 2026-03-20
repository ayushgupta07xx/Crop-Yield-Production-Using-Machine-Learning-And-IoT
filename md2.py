#!/usr/bin/env python
# coding: utf-8

# <p style = "font-size : 50px; color : #532e1c ; font-family : 'Comic Sans MS'; text-align : center; background-color : #bedcfa; border-radius: 5px 5px;"><strong>Chronic Kidney Disease Prediction</strong></p>

# <img style="margin-left: 10%; float: center;  border:5px solid #ffb037; width:80%; height : 80%;" src = https://medicaldialogues.in/h-upload/2020/12/30/145030-chronic-kidney-disease.jpg> 

# <a id = '0'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong>Table of Contents</strong></p> 
# 
# * [EDA](#2.0)
# * [Data Pre Processing](#3.0)
# * [Feature Encoding](#4.0)
# * [Model Building](#5.0)
#     * [Knn](#5.1)
#     * [Decision Tree Classifier](#5.2)
#     * [Random Forest Classifier](#5.3)
#     * [Ada Boost Classifier](#5.4)
#     * [Gradient Boosting Classifier](#5.5)
#     * [Stochastic Gradient Boosting (SGB)](#5.6)
#     * [XgBoost](#5.7)
#     * [Cat Boost Classifier](#5.8)
#     * [Extra Trees Classifier](#5.9)
#     * [LGBM Classifier](#5.10)
# 
# * [Models Comparison](#6.0)

# In[1]:


# necessary imports 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 26)


# In[2]:


# loading data

def multiple_models(dataset_path):

            df= pd.read_csv(dataset_path)

            df.drop('id', axis = 1, inplace = True)


            df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
                          'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                          'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
                          'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
                          'aanemia', 'class']



            df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
            df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
            df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')


            # Extracting categorical and numerical columns

            cat_cols = [col for col in df.columns if df[col].dtype == 'object']
            num_cols = [col for col in df.columns if df[col].dtype != 'object']


            # In[12]:


            # looking at unique values in categorical columns

            for col in cat_cols:
                print(f"{col} has {df[col].unique()} values\n")


            # <p style = "font-size : 20px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>There is some ambugity present in the columns we have to remove that.</strong></p> 

            # In[13]:


            # replace incorrect values

            df['diabetes_mellitus'].replace(to_replace = {'\tno':'no','\tyes':'yes',' yes':'yes'},inplace=True)

            df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace = '\tno', value='no')

            df['class'] = df['class'].replace(to_replace = {'ckd\t': 'ckd', 'notckd': 'not ckd'})


            # In[14]:


            df['class'] = df['class'].map({'ckd': 0, 'not ckd': 1})
            df['class'] = pd.to_numeric(df['class'], errors='coerce')


            # In[15]:


            cols = ['diabetes_mellitus', 'coronary_artery_disease', 'class']

            for col in cols:
                print(f"{col} has {df[col].unique()} values\n")







            # filling null values, we will use two methods, random sampling for higher null values and 
            # mean/mode sampling for lower null values

            def random_value_imputation(feature):
                random_sample = df[feature].dropna().sample(df[feature].isna().sum())
                random_sample.index = df[df[feature].isnull()].index
                df.loc[df[feature].isnull(), feature] = random_sample
                
            def impute_mode(feature):
                mode = df[feature].mode()[0]
                df[feature] = df[feature].fillna(mode)


            # In[52]:


            # filling num_cols null values using random sampling method

            for col in num_cols:
                random_value_imputation(col)




            # filling "red_blood_cells" and "pus_cell" using random sampling method and rest of cat_cols using mode imputation

            random_value_imputation('red_blood_cells')
            random_value_imputation('pus_cell')

            for col in cat_cols:
                impute_mode(col)





            for col in cat_cols:
                print(f"{col} has {df[col].nunique()} categories\n")


            # <p style = "font-size : 20px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>As all of the categorical columns have 2 categories we can use label encoder</strong></p> 

            # In[57]:


            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()

            for col in cat_cols:
                df[col] = le.fit_transform(df[col])





            ind_col = [col for col in df.columns if col != 'class']
            dep_col = 'class'

            X = df[ind_col]
            y = df[dep_col]


            # In[60]:


            # splitting data intp training and test set

            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


            # <a id = '5.1'></a>
            # <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>KNN</strong></p> 

            # In[61]:


            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

            knn = KNeighborsClassifier()
            knn.fit(X_train, y_train)

            # accuracy score, confusion matrix and classification report of knn

            knn_acc = accuracy_score(y_test, knn.predict(X_test))
            # Pass results to template

            print(f"Training Accuracy of KNN is {accuracy_score(y_train, knn.predict(X_train))}")
            print(f"Test Accuracy of KNN is {knn_acc} \n")

            print(f"Confusion Matrix :- \n{confusion_matrix(y_test, knn.predict(X_test))}\n")
            print(f"Classification Report :- \n {classification_report(y_test, knn.predict(X_test))}")


            # <a id = '5.2'></a>
            # <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Decision Tree Classifier</strong></p> 

            # In[62]:


            from sklearn.tree import DecisionTreeClassifier

            dtc = DecisionTreeClassifier()
            dtc.fit(X_train, y_train)

            # accuracy score, confusion matrix and classification report of decision tree

            dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

            print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, dtc.predict(X_train))}")
            print(f"Test Accuracy of Decision Tree Classifier is {dtc_acc} \n")

            print(f"Confusion Matrix :- \n{confusion_matrix(y_test, dtc.predict(X_test))}\n")
            print(f"Classification Report :- \n {classification_report(y_test, dtc.predict(X_test))}")


            # In[63]:


            # hyper parameter tuning of decision tree 

            from sklearn.model_selection import GridSearchCV
            grid_param = {
                'criterion' : ['entropy'],
                'max_depth' : [10],
                'splitter' : ['random'],
                'min_samples_leaf' : [7],
                'min_samples_split' : [7],
                'max_features' : ['sqrt']
            }

            grid_search_dtc = GridSearchCV(dtc, grid_param, cv = 5, n_jobs = -1, verbose = 1)
            grid_search_dtc.fit(X_train, y_train)


            # In[64]:


            # best parameters and best score

            print(grid_search_dtc.best_params_)
            print(grid_search_dtc.best_score_)


            # In[65]:


            # best estimator

            dtc = grid_search_dtc.best_estimator_

            # accuracy score, confusion matrix and classification report of decision tree

            dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

            print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, dtc.predict(X_train))}")
            print(f"Test Accuracy of Decision Tree Classifier is {dtc_acc} \n")

            print(f"Confusion Matrix :- \n{confusion_matrix(y_test, dtc.predict(X_test))}\n")
            print(f"Classification Report :- \n {classification_report(y_test, dtc.predict(X_test))}")


            # <a id = '5.3'></a>
            # <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Random Forest Classifier</strong></p>

            # In[66]:


            from sklearn.ensemble import RandomForestClassifier

            rd_clf = RandomForestClassifier(criterion = 'entropy', max_depth = 11, max_features = 'auto', min_samples_leaf = 2, min_samples_split = 3, n_estimators = 130)
            rd_clf.fit(X_train, y_train)

            # accuracy score, confusion matrix and classification report of random forest

            rd_clf_acc = accuracy_score(y_test, rd_clf.predict(X_test))

            print(f"Training Accuracy of Random Forest Classifier is {accuracy_score(y_train, rd_clf.predict(X_train))}")
            print(f"Test Accuracy of Random Forest Classifier is {rd_clf_acc} \n")

            print(f"Confusion Matrix :- \n{confusion_matrix(y_test, rd_clf.predict(X_test))}\n")
            print(f"Classification Report :- \n {classification_report(y_test, rd_clf.predict(X_test))}")


            # <a id = '5.4'></a>
            # <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Ada Boost Classifier</strong></p>

            # In[67]:


            from sklearn.ensemble import AdaBoostClassifier

            ada = AdaBoostClassifier(base_estimator = dtc)
            ada.fit(X_train, y_train)

            # accuracy score, confusion matrix and classification report of ada boost

            ada_acc = accuracy_score(y_test, ada.predict(X_test))

            print(f"Training Accuracy of Ada Boost Classifier is {accuracy_score(y_train, ada.predict(X_train))}")
            print(f"Test Accuracy of Ada Boost Classifier is {ada_acc} \n")

            print(f"Confusion Matrix :- \n{confusion_matrix(y_test, ada.predict(X_test))}\n")
            print(f"Classification Report :- \n {classification_report(y_test, ada.predict(X_test))}")


            # <a id = '5.5'></a>
            # <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Gradient Boosting Classifier</strong></p>

            # In[68]:


            from sklearn.ensemble import GradientBoostingClassifier

            gb = GradientBoostingClassifier()
            gb.fit(X_train, y_train)

            # accuracy score, confusion matrix and classification report of gradient boosting classifier

            gb_acc = accuracy_score(y_test, gb.predict(X_test))

            print(f"Training Accuracy of Gradient Boosting Classifier is {accuracy_score(y_train, gb.predict(X_train))}")
            print(f"Test Accuracy of Gradient Boosting Classifier is {gb_acc} \n")

            print(f"Confusion Matrix :- \n{confusion_matrix(y_test, gb.predict(X_test))}\n")
            print(f"Classification Report :- \n {classification_report(y_test, gb.predict(X_test))}")


            # <a id = '5.6'></a>
            # <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Stochastic Gradient Boosting (SGB)</strong></p>

            # In[69]:


            sgb = GradientBoostingClassifier(max_depth = 4, subsample = 0.90, max_features = 0.75, n_estimators = 200)
            sgb.fit(X_train, y_train)

            # accuracy score, confusion matrix and classification report of stochastic gradient boosting classifier

            sgb_acc = accuracy_score(y_test, sgb.predict(X_test))

            print(f"Training Accuracy of Stochastic Gradient Boosting is {accuracy_score(y_train, sgb.predict(X_train))}")
            print(f"Test Accuracy of Stochastic Gradient Boosting is {sgb_acc} \n")

            print(f"Confusion Matrix :- \n{confusion_matrix(y_test, sgb.predict(X_test))}\n")
            print(f"Classification Report :- \n {classification_report(y_test, sgb.predict(X_test))}")


            # <a id = '5.7'></a>
            # <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>XgBoost</strong></p>

            # In[70]:


            from xgboost import XGBClassifier

            xgb = XGBClassifier(objective = 'binary:logistic', learning_rate = 0.5, max_depth = 5, n_estimators = 150)
            xgb.fit(X_train, y_train)

            # accuracy score, confusion matrix and classification report of xgboost

            xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

            print(f"Training Accuracy of XgBoost is {accuracy_score(y_train, xgb.predict(X_train))}")
            print(f"Test Accuracy of XgBoost is {xgb_acc} \n")

            print(f"Confusion Matrix :- \n{confusion_matrix(y_test, xgb.predict(X_test))}\n")
            print(f"Classification Report :- \n {classification_report(y_test, xgb.predict(X_test))}")


            # <a id = '5.8'></a>
            # <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Cat Boost Classifier</strong></p>

            # In[71]:


            from catboost import CatBoostClassifier

            cat = CatBoostClassifier(iterations=10)
            cat.fit(X_train, y_train)


            # In[72]:


            # accuracy score, confusion matrix and classification report of cat boost

            cat_acc = accuracy_score(y_test, cat.predict(X_test))

            print(f"Training Accuracy of Cat Boost Classifier is {accuracy_score(y_train, cat.predict(X_train))}")
            print(f"Test Accuracy of Cat Boost Classifier is {cat_acc} \n")

            print(f"Confusion Matrix :- \n{confusion_matrix(y_test, cat.predict(X_test))}\n")
            print(f"Classification Report :- \n {classification_report(y_test, cat.predict(X_test))}")


            # <a id = '5.9'></a>
            # <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>Extra Trees Classifier</strong></p>

            # In[73]:


            from sklearn.ensemble import ExtraTreesClassifier

            etc = ExtraTreesClassifier()
            etc.fit(X_train, y_train)

            # accuracy score, confusion matrix and classification report of extra trees classifier

            etc_acc = accuracy_score(y_test, etc.predict(X_test))

            print(f"Training Accuracy of Extra Trees Classifier is {accuracy_score(y_train, etc.predict(X_train))}")
            print(f"Test Accuracy of Extra Trees Classifier is {etc_acc} \n")

            print(f"Confusion Matrix :- \n{confusion_matrix(y_test, etc.predict(X_test))}\n")
            print(f"Classification Report :- \n {classification_report(y_test, etc.predict(X_test))}")


            # <a id = '5.10'></a>
            # <p style = "font-size : 25px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #fbc6a4; border-radius: 5px 5px;"><strong>LGBM Classifier</strong></p>

            # In[74]:


            from lightgbm import LGBMClassifier

            lgbm = LGBMClassifier(learning_rate = 1)
            lgbm.fit(X_train, y_train)

            # accuracy score, confusion matrix and classification report of lgbm classifier

            lgbm_acc = accuracy_score(y_test, lgbm.predict(X_test))

            print(f"Training Accuracy of LGBM Classifier is {accuracy_score(y_train, lgbm.predict(X_train))}")
            print(f"Test Accuracy of LGBM Classifier is {lgbm_acc} \n")

            print(f"{confusion_matrix(y_test, lgbm.predict(X_test))}\n")
            print(classification_report(y_test, lgbm.predict(X_test)))


            # <a id = '6.0'></a>
            # <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong>Models Comparison</strong></p> 

            # In[75]:


            models = pd.DataFrame({
                'Model' : [ 'KNN', 'Decision Tree Classifier', 'Random Forest Classifier','Ada Boost Classifier',
                         'Gradient Boosting Classifier', 'Stochastic Gradient Boosting', 'XgBoost', 'Cat Boost', 'Extra Trees Classifier'],
                'Score' : [knn_acc, dtc_acc, rd_clf_acc, ada_acc, gb_acc, sgb_acc, xgb_acc, cat_acc, etc_acc]
            })


            models.sort_values(by = 'Score', ascending = False)


            # In[76]:


            px.bar(data_frame = models, x = 'Score', y = 'Model', color = 'Score', template = 'plotly_dark', 
                   title = 'Models Comparison')


            # <p style = "font-size : 25px; color : #f55c47 ; font-family : 'Comic Sans MS'; "><strong>If you like my work, don't forget to leave an upvote!!</strong></p> 

            # In[ ]:





# In[ ]:




