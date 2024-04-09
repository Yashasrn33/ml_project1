# Internal Status Prediction API

This project implements a machine learning model to predict the internal status of a process based on its external status descriptions. The model is exposed through a FastAPI-based API, allowing users to make predictions using external status descriptions.

## Overview

The project consists of the following components:

1. **Data Preprocessing**: The provided dataset is preprocessed to clean and format the external status descriptions and internal status labels for training the machine learning model.

2. **Model Development**: A machine learning model is developed using TensorFlow to predict the internal status based on external status descriptions. The model architecture includes LSTM layers to capture sequential patterns in the input text data.

3. **Model Training and Evaluation**: The developed model is trained on the preprocessed dataset and evaluated using metrics such as accuracy, precision, recall, and F1-score.

4. **API Development**: An API is implemented using FastAPI framework to expose the trained machine learning model. The API accepts external status descriptions as input and returns predicted internal status labels.

5. **Testing and Validation**: The developed API is thoroughly tested to ensure its functionality and accuracy. Predictions are validated against a validation dataset to measure the model's generalization ability.

## Usage

### Prerequisites

- Python 3.6 or higher
- TensorFlow
- FastAPI
- Pydantic
- scikit-learn

### Running the API

1. Upload the folder onto VScode, n run the code "uvicorn main:app" in the terminal

2. Access the API documentation and test the endpoints by visiting http://localhost:8000/docs in your browser.
