# 📈 Financial Time Series Prediction with RNN 


![Sans titre](https://github.com/user-attachments/assets/665b99d5-f3b6-4a74-9d8a-48434495911f)
![Sans titre](https://github.com/user-attachments/assets/ad3de3c8-6297-4a13-83e1-963c2d848420)


# RNN Finance Project

## Description
This project implements a Recurrent Neural Network (RNN) model applied to financial analysis. The goal is to predict financial trends based on time-series data using advanced deep learning techniques.

## Technologies Used
- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/DOHA-IDRISSI-MOUNADI/rnn_finance_project.git
   ```
2. Navigate to the project folder:
   ```bash
   cd rnn_finance_project
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Open the Jupyter notebook:
   ```bash
   jupyter notebook rnn_FINANCE_project.ipynb
   ```
2. Run the notebook cells to train and test the model.

## Code Snippets
Here are some key code snippets from the project:

### Data Preprocessing
```python
import pandas as pd
import numpy as np

data = pd.read_csv("financial_data.csv")
data = data.fillna(method='ffill')
print(data.head())
```

### Model Architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
```

### Training the Model
```python
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
```

## Results
Below are the results of the latest models tested:

### Model Performance
You can insert performance metrics such as loss, accuracy, RMSE, etc.
```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

### Predictions vs. Actual Data
import matplotlib.pyplot as plt
import numpy as np

# Set the size of the plot
plt.rcParams['figure.figsize'] = [14, 4]

# Test data
plt.plot(test_data.index[-100:-30], test_data.Open[-100:-30], label="test_data", color="b")

# Reverse the scaling transformation
original_cases = scaler.inverse_transform(np.expand_dims(sequence_to_plot[-1], axis=0)).flatten()

# The historical data used as input for forecasting
plt.plot(test_data.index[-30:], original_cases, label='actual values', color='green')

# Forecasted Values
# Reverse the scaling transformation
forecasted_cases = scaler.inverse_transform(np.expand_dims(forecasted_values, axis=0)).flatten()

# Ensure combined_index is aligned with forecasted_cases
aligned_combined_index = combined_index[-len(forecasted_cases):]

# Plotting the forecasted values
plt.plot(aligned_combined_index, forecasted_cases, label='forecasted values', color='red')


![Sans titre](https://github.com/user-attachments/assets/15209b42-0f85-4f69-86e7-d019397daa4e)



## Objectives
- Apply artificial intelligence to financial analysis.
- Experiment with recurrent neural networks (RNN, LSTM, GRU) on financial time series.
- Understand the impact of model parameters on prediction performance.

## Author
Doha Idrissi Mounadi - Data Engineer | Master's in Spatial Analysis and IT Development



## Contact
- Email: dohaidrissimou@gmail.com
- LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/your-profile)
- GitHub: [DOHA IDRISSI MOUNADI](https://github.com/DOHA-IDRISSI-MOUNADI)

---




