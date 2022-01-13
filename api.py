from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import uvicorn

###############################
# Model

# Loading Iris Dataset
iris = load_iris()

# Getting features and targets from dataset
X = iris.data
y = iris.target

# Fitting our Model on the dataset
clf = GaussianNB()
clf.fit(X, y)

###############################
# API

# Declaring our FastAPI instance
app = FastAPI()

# Data model for pydantic
class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

@app.post('/predict')
def predict(data : request_body):
    test_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    class_idx = clf.predict(test_data)[0]
    return { 'class' : iris.target_names[class_idx]}


# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Welcome to my App!'}

# Defining path operation for /name endpoint
@app.get('/{name}')
def hello_name(name : str):
    # Defining a function that takes only string as input and output the
    # following message.
    return {'message': f'Weldome to my app, {name}'}