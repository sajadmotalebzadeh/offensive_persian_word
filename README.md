
# 🛡️ Persian Offensive Word Detection with Deep Learning

This repository contains a complete project for detecting offensive and abusive language in Persian (Farsi) text using Natural Language Processing (NLP) and Deep Learning. The goal is to help moderate Persian-language content and maintain respectful communication in online platforms.

## 🔍 Project Overview

This project leverages an **LSTM-based neural network** trained on the **PHATE dataset**, a well-known Persian dataset for offensive language detection. The model classifies Persian sentences or phrases as **offensive** or **non-offensive** with high accuracy.

## 🚀 Features

* ✅ Deep learning model (LSTM) trained on labeled Persian text
* ✅ Preprocessing pipeline for Persian tokenization and normalization
* ✅ REST API implementation with **microservices architecture**
* ✅ API Key authentication system for secure access
* ✅ Web interface built with **Streamlit** for demonstration and testing
* ✅ Deployable as a standalone service

## 🧠 Model

* Architecture: LSTM (Long Short-Term Memory)
* Framework: TensorFlow / Keras
* Model file: `TF_model.h5`
* Input: Raw Persian text
* Output: Binary classification (`offensive`, `non-offensive`)

## 📦 Technologies Used

* Python
* TensorFlow / Keras
* NLTK & Hazm for Persian text preprocessing
* FastAPI (for API)
* Streamlit (for demo UI)
* Docker (optional for containerization)

## 🌐 Live Demo

Coming Soon – You can run the project locally using Streamlit to test the model.

## 🔐 API Access

The REST API is protected with API Key authentication. You must provide a valid key in the request headers to access the endpoints.

## 📁 Folder Structure

```
├── api/                  # FastAPI backend
├── streamlit_app/        # Streamlit frontend
├── model/                # Saved model (TF_model.h5)
├── data/                 # Dataset and preprocessing scripts
├── requirements.txt
└── README.md
```

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score

## 🤝 Contribution

Contributions are welcome! Feel free to submit issues or pull requests if you'd like to improve the model or enhance the API.

## 📜 License

This project is licensed under the MIT License.
