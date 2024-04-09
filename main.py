import json
import re
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Load dataset
with open('dataset.json', 'r') as f:
    data = json.load(f)

# Load external status and internal status from dataset
external_status = [record['externalStatus'] for record in data]
internal_status = [record['internalStatus'] for record in data]

# Data cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply data cleaning
external_status = [clean_text(text) for text in external_status]

# Encode labels
label_encoder = LabelEncoder()
encoded_internal_status = label_encoder.fit_transform(internal_status)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(external_status, encoded_internal_status, test_size=0.2, random_state=42)

# Tokenize and pad sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X_train)
max_seq_length = 100
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=max_seq_length)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=max_seq_length)

# Define LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_seq_length),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Define FastAPI app
app = FastAPI()

# Define request model
class StatusRequest(BaseModel):
    externalStatus: str

# Define response model
class StatusResponse(BaseModel):
    internalStatus: str

# Define API endpoint for status prediction
@app.post("/predict", response_model=StatusResponse)
async def predict_status(status: StatusRequest):
    sequence = tokenizer.texts_to_sequences([status.externalStatus])
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_seq_length)
    prediction = model.predict(sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return StatusResponse(internalStatus=predicted_label)

# Define endpoint to serve HTML content
@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)
