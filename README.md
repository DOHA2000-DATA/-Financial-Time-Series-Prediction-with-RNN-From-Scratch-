# 📈 Financial Time Series Prediction with RNN (From Scratch) 


![Sans titre](https://github.com/user-attachments/assets/665b99d5-f3b6-4a74-9d8a-48434495911f)
![Sans titre](https://github.com/user-attachments/assets/ad3de3c8-6297-4a13-83e1-963c2d848420)


## Description

This project implements a Recurrent Neural Network (RNN) from scratch, without using advanced frameworks like TensorFlow or PyTorch. The primary objective is to explore and understand the core principles of RNNs, such as forward propagation, backpropagation through time (BPTT), and sequential learning, while applying them to financial time series prediction.

By training the model on historical financial data, we aim to analyze how well an RNN can capture trends and predict future values based on past sequences.

# # Project Objectives


🔹 Develop an RNN from scratch using NumPy to fully understand its inner workings.

🔹 Train the model on financial time series data (e.g., stock prices, exchange rates).

🔹 Learn how RNNs handle sequential data and compare performance over different time windows.

🔹 Experiment with different hyperparameters and evaluate their impact on prediction accuracy.

🔹 Visualize the model’s learning process and assess potential improvements

# Methodology

1️⃣ Data Collection & Preprocessing 📊

    Load financial time series data (stock prices, forex rates, etc.).
    Normalize the data and convert it into time-step sequences for training.

2️⃣ RNN Model Implementation 🧠

    Build a simple RNN model from scratch using NumPy.
    Implement forward propagation and backpropagation through time (BPTT).

3️⃣ Model Training & Optimization 🚀

    Train the RNN on historical financial data.
    Adjust hyperparameters (learning rate, hidden units, sequence length).

4️⃣ Predictions & Evaluation 📈

    Use the trained model to predict future values.
    Visualize and compare real vs predicted data.

5️⃣ Analysis & Future Improvements 🔬

    Evaluate performance and discuss potential enhancements (LSTM, GRU, advanced preprocessing).

## Sample Predictions
Date	Actual Price	Predicted Price
2025-01-01	$100.5	$101.2
2025-01-02	$101.8	$102.0
2025-01-03	$102.3	$101.9
2025-01-04	$101.5	$101.6
🔧 Installation & Execution

1️⃣ Clone the repository

git clone https://github.com/idrissi-mounadi-doha/RNN_FINANCE_PROJECT.git
cd RNN_FINANCE_PROJECT

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Train the RNN model

python src/train.py

4️⃣ Make predictions using the trained model

python src/predict.py

👨‍💻 Author


🔹 IDRISSI MOUNADI DOHA

📧 Email: dohaidrissimou@gmail.com

🔗 LinkedIn: Doha Idrissi Mounadi


