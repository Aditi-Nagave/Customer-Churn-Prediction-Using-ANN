# Customer Churn Prediction Using ANN

A Deep Learning web application that predicts whether a customer is likely to churn or not using an Artificial Neural Network (ANN).

## Project Overview

This project uses:

- TensorFlow / Keras for building the ANN model
- Scikit-learn for preprocessing
- Streamlit for the web interface
- Pandas and NumPy for data handling

The model predicts customer churn based on customer details like:

- Credit Score
- Geography
- Gender
- Age
- Balance
- Number of Products
- Active Member Status
- Estimated Salary
- Tenure

---

## Tech Stack

- Python
- TensorFlow
- Keras
- Streamlit
- Scikit-learn
- Pandas
- NumPy

---

## Project Structure

```bash
Customer-Churn-Prediction-Using-ANN/
│
├── app.py
├── model.h5
├── scaler.pkl
├── one_hot_encoder.pkl
├── label_encoder_gender.pkl
├── requirements.txt
├── README.md
└── .gitignore


## Installation

Clone the repository:

```bash
git clone https://github.com/Aditi-Nagave/Customer-Churn-Prediction-Using-ANN.git
```

Move to project directory:

```bash
cd Customer-Churn-Prediction-Using-ANN
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Features

- Interactive Streamlit UI
- Real-time churn prediction
- One-hot encoding and label encoding
- Feature scaling using StandardScaler
- Deep learning prediction using ANN

---

## Model Information

The ANN model was trained using:

- Dense Neural Network Layers
- Sigmoid Output Activation
- Binary Crossentropy Loss
- Adam Optimizer

---

## Deployment

This project can be deployed using:

- Streamlit Community Cloud
- GitHub

---

## Author

Aditi Nagave
