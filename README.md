LSTM-Based QSP Model
This project uses an LSTM neural network to learn temporal patterns from Quantitative Systems Pharmacology (QSP) data. It uses 11 key input parameters to predict a 200-dimensional output vector, which could represent a dynamic pharmacokinetic or pharmacodynamic response profile over time.

🧠 Objective
Train a multi-layer LSTM network on QSP simulation data using:

11 input features (physiological/pathophysiological parameters)

200 output targets (e.g., time series of drug concentration/effect)

Min-Max scaling

MAE loss

📁 Data Format
Input CSV file should contain at least 211 columns:

Columns 0–10: input features (shape: (4999, 11))

Columns 11–210: output variables (shape: (4999, 200))

Last column: optional validation tag (shape: (4999, 1))

🔄 Preprocessing
python
Copy
Edit
x_df = np.array(data.iloc[:, 0:11])
y_df = np.array(data.iloc[:, 11:211])
valid_df = np.array(data.iloc[:, -2:-1])

# Normalize and reshape inputs for LSTM
x_df = scaler.fit_transform(x_df)
x_df = np.reshape(x_df, (4999, 11, 1))
🧪 Train-Test Split
python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.01, random_state=42)
🧱 Model Architecture
python
Copy
Edit
model = Sequential()
model.add(LSTM(128, batch_input_shape=(None, 11, 1), return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(200, return_sequences=False))
model.add(Dense(200))
model.compile(loss='mean_absolute_error', optimizer='adam')
Input shape: (11 timesteps, 1 feature)

Output shape: (200,)

🚀 Training
python
Copy
Edit
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=500,
    batch_size=32,
    verbose=2
)
Loss: Mean Absolute Error (MAE)

Optimizer: Adam

Epochs: 500

Batch size: 32

📊 Outputs
After training, the model can be used to predict a 200-dimensional time profile for new QSP parameter configurations.

📌 Requirements
Python 3.7+

TensorFlow / Keras

NumPy, Pandas, scikit-learn

Install dependencies:

bash
Copy
Edit
pip install numpy pandas scikit-learn tensorflow
