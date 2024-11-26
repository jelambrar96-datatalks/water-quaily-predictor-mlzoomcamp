#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os


# In[8]:


import requests


# In[9]:


import numpy as np
import pandas as  pd


# In[10]:


import warnings
warnings.filterwarnings('ignore')


# ## 2. Load Dataset

# In[11]:


URL = "https://raw.githubusercontent.com/Sarthak-1408/Water-Potability/refs/heads/main/water_potability.csv"
DATASET_FILEPATH = "./water_potability.csv"
if not os.path.isfile(DATASET_FILEPATH):
    response = requests.get(URL)
    # Check if the download was successful
    if response.status_code == 200:
        with open('water_potability.csv', 'wb') as file:
            file.write(response.content)
        print("CSV file downloaded successfully.")
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

df = pd.read_csv(DATASET_FILEPATH)


# In[12]:


# df = df.sample(100, random_state=1)


# ## 3. Transform Columns

# In[13]:


df.columns = df.columns.str.lower()


# In[14]:


COLUMNS = list(df.columns)
print(COLUMNS)


# In[15]:


TARGET_COLUMN = 'potability'
COLUMNS.remove(TARGET_COLUMN)


# In[16]:


df.head(5)


# In[17]:


df.info()


# ## 3. Remove nulls

# In[18]:


df.fillna(0, inplace=True)


# ## 4. Create Dataset

# In[19]:


from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = (df_train[TARGET_COLUMN]).astype('int').values
y_val = (df_val[TARGET_COLUMN]).astype('int').values
y_test = (df_test[TARGET_COLUMN]).astype('int').values

del df_train[TARGET_COLUMN]
del df_val[TARGET_COLUMN]
del df_test[TARGET_COLUMN]


# In[20]:


from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)


# ### 4.1 Save pickle
# 

# In[69]:


import pickle
from typing import Any, Optional
from pathlib import Path
import logging
import os


def save_to_pickle(
    obj: Any,
    file_path: str,
    create_dir: bool = True,
    compression: Optional[str] = None
) -> bool:
    """
    Save an object to a pickle file with error handling.
    
    Parameters:
    -----------
    obj : Any
        The Python object to save
    file_path : str
        Path where the pickle file will be saved
    create_dir : bool, optional (default=True)
        If True, creates the directory if it doesn't exist
    compression : str, optional (default=None)
        Compression protocol to use ('gzip', 'bz2', 'lzma' or None)
        
    Returns:
    --------
    bool
        True if save was successful, False otherwise
        
    Examples:
    --------
    >>> data = {'key': 'value'}
    >>> save_to_pickle(data, 'data/my_dict.pkl')
    >>> save_to_pickle(data, 'data/my_dict.pkl.gz', compression='gzip')
    """
    try:
        # Convert to Path object for better path handling
        path = Path(file_path)
        
        # Create directory if it doesn't exist and create_dir is True
        if create_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
            
        # Determine the appropriate open function and mode
        if compression:
            if compression == 'gzip':
                import gzip
                open_func = gzip.open
            elif compression == 'bz2':
                import bz2
                open_func = bz2.open
            elif compression == 'lzma':
                import lzma
                open_func = lzma.open
            else:
                raise ValueError(f"Unsupported compression format: {compression}")
            mode = 'wb'
        else:
            open_func = open
            mode = 'wb'

        # Save the object
        with open_func(path, mode) as f:
            pickle.dump(obj, f)
        return True

    except Exception as e:
        print(e)
        return False


# In[ ]:


save_to_pickle(dv, './dv.pkl')


# ## 5. Trainnning models
# 
# ### 5.0. Utils functions

# In[21]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# In[22]:


# Detalles adicionales de evaluación
from sklearn.metrics import (
    classification_report, 
    confusion_matrix
)


# In[23]:


def evaluate_model(model, X_train, y_train, X_val, y_val, params):
    """
    # Función para evaluar un conjunto de hiperparámetros
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', model(**params))
    ])    
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    return accuracy, pipeline


# In[24]:


from itertools import product

def find_best_model(Model, parameter_grid, X_train, y_train, X_val, y_val, extra_parameters = {}, verbose = False):
    best_accuracy = -np.inf
    best_params = None
    best_model = None

    parameter_labels = parameter_grid.keys()
    parameter_values = parameter_grid.values()

    for temp_parameter_iterable in product(*parameter_values):
        params = { label:value for label, value in zip(parameter_labels, temp_parameter_iterable) }
        if verbose:
            print()
            print(params)

        # Evaluamos los parámetros
        try:
            accuracy, model = evaluate_model(
                Model, X_train, y_train, X_val, y_val, params
            )
        except ValueError as ve:
            if verbose:
                print(ve)
            continue

        if verbose:
            print(f"accuracy: {accuracy}")
        
        # Actualizamos mejor modelo si es necesario
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_model = model

    return best_model, best_params, best_accuracy


# ### 5.1. Configuring MLFLOW

# In[25]:


"""
from mlflow.tracking import MlflowClient

TRACKING_SERVER_HOST = "localhost"
client = MlflowClient(f"http://{TRACKING_SERVER_HOST}:5000")

mlflow.set_experiment("wqm-exp-1")
"""


# ### 5.2 Model 1: Logistic regression 

# In[26]:


from sklearn.linear_model import LogisticRegression

param_grid = {
    'penalty':['l1','l2','elasticnet', None],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter'  : [100,1000,2500,5000]
}

best_model, best_params, val_accuracy = find_best_model(
    Model=LogisticRegression,
    parameter_grid=param_grid,
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,)


# In[27]:


# Imprimir resultados de optimización
print("Mejores Hiperparámetros:")
for param, value in best_params.items():
    print(f"{param}: {value}")
print(f"\nAccuracy en Validación: {val_accuracy:.4f}")


# In[28]:


# Evaluar en conjunto de prueba
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy en Prueba: {test_accuracy:.4f}")


# In[29]:


print("\nInforme de Clasificación:")
print(classification_report(y_test, y_test_pred))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_test_pred))


# In[ ]:


save_to_pickle(best_model, './model_logistic_regression.pkl')


# ### 5.3 Model 2: Random Forest

# In[30]:


# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'random_state': [1]
}


# In[31]:


from sklearn.ensemble import RandomForestClassifier

best_model, best_params, val_accuracy = find_best_model(
    Model=RandomForestClassifier,
    parameter_grid=param_grid,
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,)


# In[32]:


# Evaluar en conjunto de prueba
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy en Prueba: {test_accuracy:.4f}")


# In[33]:


print("\nInforme de Clasificación:")
print(classification_report(y_test, y_test_pred))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_test_pred))


# In[ ]:


save_to_pickle(best_model, './model_random_forest.pkl')


# ### 5.4. Model 3: Support Vector Machine

# In[34]:


param_grid = {
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
    'degree': [2, 3, 4],  # only relevant for poly kernel
    'class_weight': [None, 'balanced'],
    'decision_function_shape': ['ovr', 'ovo']
}


# In[35]:


from sklearn.svm import SVC

best_model, best_params, val_accuracy = find_best_model(
    Model=SVC,
    parameter_grid=param_grid,
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,verbose=False)


# In[36]:


# Evaluar en conjunto de prueba
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy en Prueba: {test_accuracy:.4f}")


# In[37]:


print("\nInforme de Clasificación:")
print(classification_report(y_test, y_test_pred))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_test_pred))


# In[39]:


param_grid = {
    'kernel': ['poly'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
    'degree': [2, 3, 4],  # only relevant for poly kernel
    'class_weight': [None, 'balanced'],
    'decision_function_shape': ['ovr', 'ovo']
}


# In[40]:


best_model, best_params, val_accuracy = find_best_model(
    Model=SVC,
    parameter_grid=param_grid,
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,)


# In[41]:


# Evaluar en conjunto de prueba
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy en Prueba: {test_accuracy:.4f}")


# In[42]:


print("\nInforme de Clasificación:")
print(classification_report(y_test, y_test_pred))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_test_pred))


# In[ ]:


save_to_pickle(best_model, './model_svm_forest.pkl')


# ### 5.5 Model 4: Native Bayes 

# In[43]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


# ### 5.5.1 Gaussian Naive Bayes

# In[44]:


param_grid = {
    'var_smoothing': np.logspace(-11, -1, 20),
    'priors': [None]  # Can be extended with specific priors if needed
}


# In[45]:


best_model, best_params, val_accuracy = find_best_model(
    Model=GaussianNB,
    parameter_grid=param_grid,
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,)


# In[46]:


# Evaluar en conjunto de prueba
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy en Prueba: {test_accuracy:.4f}")


# In[47]:


print("\nInforme de Clasificación:")
print(classification_report(y_test, y_test_pred))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_test_pred))


# In[ ]:


save_to_pickle(best_model, './model_gaussian_nb.pkl')


# ### 5.5.1 Bernoulli Naive Bayes

# In[57]:


param_grid = {
    'alpha': np.logspace(-3, 3, 20),
    'binarize': [None, 0.0, 0.25, 0.5, 0.75, 1.0],  # Threshold for binarizing features
    'fit_prior': [True, False]
}


# In[59]:


best_model, best_params, val_accuracy = find_best_model(
    Model=BernoulliNB,
    parameter_grid=param_grid,
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,)


# In[61]:


# Evaluar en conjunto de prueba
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy en Prueba: {test_accuracy:.4f}")


# In[62]:


print("\nInforme de Clasificación:")
print(classification_report(y_test, y_test_pred))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_test_pred))


# In[72]:


save_to_pickle(best_model, './model_bernoulli_nb.pkl')

