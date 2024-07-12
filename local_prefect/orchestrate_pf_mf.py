#import pathlib
#import pickle
import pandas as pd
import numpy as np
#import scipy
import sklearn
from typing import List

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error

from typing import List

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
from prefect import flow, task

import mlflow
#import xgboost as xgb

# 'data/online_gaming_behavior_dataset.csv'
@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_csv(filename)
    print("INFO: Data readed")
    print("Dataset size: ", df.shape)
    return df

@task
def prepare_date(df: pd.DataFrame) -> pd.DataFrame:

    df['InGamePurchases'] = df['InGamePurchases'].map({0:'No',1:'Yes'})
    #df['Age'] = pd.cut(df['Age'], bins=[15,25,35,49], labels=['Teen','Adult','Old'])
    df['EngagementLevel'] = df['EngagementLevel'].map({'Low':0,'Medium':1,'High':2})

    #df = df.astype({'InGamePurchases':'category','GameDifficulty':'category'})

    df = df.drop(columns=['PlayerID'])
    
    print("INFO: Data preprocessed")
    print("Dataset size: ", df.shape)

    return df

@task(log_prints=True)
def split_data(df:  pd.DataFrame, test_size: float = 0.2) -> tuple ([
     pd.DataFrame,
     np.ndarray,
     pd.DataFrame,
     np.ndarray ]):
    
    X = df.drop(columns=['EngagementLevel'])
    y = df.EngagementLevel

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,stratify=y,shuffle=True,random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size,stratify=y_train,shuffle=True,random_state=42)

    print("X train: ", X_train.shape)
    print("y train: ", y_train.shape)
    print("X test: ", X_test.shape)
    print("t test: ", y_test.shape)

    return X_train, X_test, y_train, y_test

@task
def build_model_pipeline(X: pd.DataFrame):
    
    # Define the types of columns
    numeric_col = X.select_dtypes(exclude=['category','object']).columns.tolist()
    #categorical_col= ['Age', 'Gender', 'Location', 'GameGenre', 'InGamePurchases', 'GameDifficulty']
    #categorical_col= ['Gender', 'Location', 'GameGenre', 'InGamePurchases', 'GameDifficulty']
    categorical_col= X.select_dtypes(include=['category','object']).columns.tolist()
    
    # Make Pipeline
    # Define pipeline for numeric columns
    num = Pipeline([
        ('imp',SimpleImputer(strategy='mean')),
        ('scl',StandardScaler())
    ])
    #  Define pipeline for categorical columns
    cat = Pipeline([
        ('imp',SimpleImputer(strategy='most_frequent')),
        ('enc',OneHotEncoder())
    ])
    # DEfine the transformer
    preprocessor = ColumnTransformer([
        ('num',num,numeric_col),
        ('cat',cat,categorical_col)
    ])    
    
    model= GradientBoostingClassifier()
    
    pipeline = Pipeline([
        ('prep',preprocessor),
        ('model',model)
    ])
    
    print("Pipeline created")
    
    return pipeline

@task(log_prints=True)
def train_evaluate_model(pipeline: sklearn.pipeline, X_train: pd.DataFrame, y_train: np.ndarray,
                         X_test: pd.DataFrame, y_test: np.ndarray,
                         ):
    
    mlflow.set_tracking_uri("http://localhost:5005")
    mlflow.set_experiment("online_gaming_1")
    
    mlflow.autolog()
    
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred,average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
    print("Results: ", [accuracy, precision, recall, f1])
    
    mlflow.sklearn.log_model(pipeline, artifact_path="models")
    
    return pipeline


@flow
def main_flow(
    train_path: str = "./data/online_gaming_behavior_dataset.csv",
) -> None:
    """The main training pipeline"""

    # MLflow settings
    #mlflow.set_tracking_uri("http://localhost:5005")
    #mlflow.set_experiment("online_gaming")


    # Load
    df_train = read_data(train_path)
    #df_val = read_data(val_path)
    print(df_train.head(5))
    # Transform
    df_train= prepare_date(df_train)
    print(df_train.head(5))
    # Split data
    X_train, X_test, y_train, y_test= split_data(df_train,test_size= 0.2)
    
    # Create pipeline
    pipeline= build_model_pipeline(X_train)
    pipeline= train_evaluate_model(pipeline, X_train, y_train,X_test, y_test)
    
def test_s3_store():
    features = "rooms, zipcode, median_price, school_rating, transport"
    with open("features.txt", 'w') as f:
        f.write(features)
    mlflow.set_tracking_uri("http://localhost:5005")
    mlflow.set_experiment("test_s3_store_2")

    with mlflow.start_run():
        art_uri=mlflow.get_artifact_uri()
        print("Artifact URI: ", art_uri)
        mlflow.log_param("p", 'p')
        mlflow.log_metric("m", 1.5)
        mlflow.log_artifact("features.txt")


if __name__ == "__main__":
    #main_flow.serve(name="full-ol-gaming-dp")    
    main_flow()
    #test_s3_store()
