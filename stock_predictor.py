import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("/content/ADANIPORTS (1).csv")
print(df.head())

data = df['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

sequence_length = 60
X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(14, 6))
plt.plot(actual_prices, color='blue', label='Actual Price')
plt.plot(predicted_prices, color='red', label='Predicted Price')
plt.title(f' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

pip install ta

import ta

ticker = "/content/ADANIPORTS (1).csv"
df = pd.read_csv(ticker)

df = df.dropna().reset_index(drop=True)

close_prices = df['Close']

df['SMA_20'] = ta.trend.sma_indicator(close_prices, window=20)
df['RSI'] = ta.momentum.rsi(close_prices, window=14)
df['MACD'] = ta.trend.macd_diff(close_prices)

df.dropna(inplace=True)

print(df[['Close', 'SMA_20', 'RSI', 'MACD']].head())

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

features = ['Close', 'SMA_20', 'RSI', 'MACD']
data = df[features]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

def create_sequences(dataset, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(dataset)):
        X.append(dataset[i-seq_length:i])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X_seq, y_seq = create_sequences(scaled, seq_length)

split = int(0.8 * len(X_seq))
X_lstm_train, X_lstm_test = X_seq[:split], X_seq[split:]
y_lstm_train, y_lstm_test = y_seq[:split], y_seq[split:]

X_flat = scaled[seq_length:]
y_flat = scaled[seq_length:, 0]

X_train, X_test, y_train, y_test = train_test_split(X_flat, y_flat, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2])),
    LSTM(50),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_lstm_train, y_lstm_train, epochs=10, batch_size=32)

# Predict
lstm_pred = model_lstm.predict(X_lstm_test)
lstm_mse = mean_squared_error(y_lstm_test, lstm_pred)

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

model_xgb = XGBRegressor()
model_xgb.fit(X_train, y_train)
xgb_pred = model_xgb.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_pred)

from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
rf_pred = model_rf.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)

print(f"LSTM MSE: {lstm_mse:.6f}")
print(f"XGBoost MSE: {xgb_mse:.6f}")
print(f"Random Forest MSE: {rf_mse:.6f}")

import matplotlib.pyplot as plt

y_lstm_test_inv = scaler.inverse_transform(np.concatenate([y_lstm_test.reshape(-1, 1),
                                                           np.zeros((len(y_lstm_test), len(features) - 1))], axis=1))[:, 0]
lstm_pred_inv = scaler.inverse_transform(np.concatenate([lstm_pred,
                                                         np.zeros((len(lstm_pred), len(features) - 1))], axis=1))[:, 0]

y_test_inv = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1),
                                                      np.zeros((len(y_test), len(features) - 1))], axis=1))[:, 0]
xgb_pred_inv = scaler.inverse_transform(np.concatenate([xgb_pred.reshape(-1, 1),
                                                        np.zeros((len(xgb_pred), len(features) - 1))], axis=1))[:, 0]
rf_pred_inv = scaler.inverse_transform(np.concatenate([rf_pred.reshape(-1, 1),
                                                       np.zeros((len(rf_pred), len(features) - 1))], axis=1))[:, 0]

plt.figure(figsize=(15, 6))
plt.plot(y_lstm_test_inv, label="Actual (LSTM)", color='blue', alpha=0.6)
plt.plot(lstm_pred_inv, label="Predicted (LSTM)", color='red')
plt.title("LSTM Prediction vs Actual")
plt.legend()
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(y_test_inv, label="Actual (XGBoost)", color='blue', alpha=0.6)
plt.plot(xgb_pred_inv, label="Predicted (XGBoost)", color='green')
plt.title("XGBoost Prediction vs Actual")
plt.legend()
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(y_test_inv, label="Actual (Random Forest)", color='blue', alpha=0.6)
plt.plot(rf_pred_inv, label="Predicted (Random Forest)", color='purple')
plt.title("Random Forest Prediction vs Actual")
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mae_lstm = mean_absolute_error(y_lstm_test, lstm_pred)
r2_lstm = r2_score(y_lstm_test, lstm_pred)

mae_xgb = mean_absolute_error(y_test, xgb_pred)
r2_xgb = r2_score(y_test, xgb_pred)

mae_rf = mean_absolute_error(y_test, rf_pred)
r2_rf = r2_score(y_test, rf_pred)

import pandas as pd

results = pd.DataFrame({
    'Model': ['LSTM', 'XGBoost', 'Random Forest'],
    'MSE': [lstm_mse, xgb_mse, rf_mse],
    'RMSE': [np.sqrt(lstm_mse), np.sqrt(xgb_mse), np.sqrt(rf_mse)],
    'MAE': [mae_lstm, mae_xgb, mae_rf],
    'R² Score': [r2_lstm, r2_xgb, r2_rf]
})

results = results.round(4)

print("Enhanced Model Comparison Table:\n")
print(results)

df['Future_Close'] = df['Close'].shift(-1)
df['Price_Change'] = (df['Future_Close'] - df['Close']) / df['Close']

df['Signal'] = np.select(
    [df['Price_Change'] > 0.01, df['Price_Change'] < -0.01],
    [1, -1],
    default=0
)

df.dropna(inplace=True)

initial_cash = 10000
cash = initial_cash
position = 0
portfolio_values = []

for i, row in df.iterrows():
    price = row['Close']
    signal = row['Signal']

    if signal == 1 and cash >= price:
        position = cash // price
        cash -= position * price

    elif signal == -1 and position > 0:
        cash += position * price
        position = 0

    portfolio_value = cash + (position * price)
    portfolio_values.append(portfolio_value)

df['Portfolio_Value'] = portfolio_values

returns = pd.Series(portfolio_values).pct_change().dropna()
total_return = (portfolio_values[-1] - initial_cash) / initial_cash
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

print(f"Total Return: {total_return*100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['Portfolio_Value'], label='Strategy Portfolio Value', color='blue')
plt.title("Strategy Simulation: Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'SMA_20': np.random.randn(n_samples),
    'RSI': np.random.uniform(10, 90, size=n_samples),
    'MACD': np.random.randn(n_samples),
    'Volume': np.random.rand(n_samples) * 1000,
    'Signal': np.random.choice([-1, 0, 1], size=n_samples, p=[0.3, 0.4, 0.3])  # Labels
})

X = df.drop('Signal', axis=1)
y = df['Signal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year

avg_close_per_year = df.groupby('Year')['Close'].mean()

plt.figure(figsize=(12, 6))
avg_close_per_year.plot(kind='bar', color='skyblue')
plt.title("Average Close Price Per Year")
plt.ylabel("Average Close Price")
plt.xlabel("Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', color='darkgreen')
plt.title("Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)
plt.tight_layout()
plt.show()

median_deliverable = df['%Deliverble'].median()
above_median = (df['%Deliverble'] > median_deliverable).sum()
below_median = (df['%Deliverble'] <= median_deliverable).sum()

plt.figure(figsize=(6, 6))
plt.pie([above_median, below_median], labels=['Above Median', 'Below Median'], autopct='%1.1f%%',
        colors=['#66b3ff', '#ff9999'], startangle=140)
plt.title("Distribution of %Deliverble")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
sns.boxplot(x='Year', y='Volume', data=df)
plt.title("Volume Distribution Per Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import shap

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

!pip install lime

import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=["Sell", "Hold", "Buy"],
    mode='classification'
)

i = 1
exp = explainer.explain_instance(X_test.iloc[i].values, clf.predict_proba)
exp.show_in_notebook(show_all=False)

np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({
    'SMA_20': np.random.randn(n_samples),
    'RSI': np.random.uniform(10, 90, size=n_samples),
    'MACD': np.random.randn(n_samples),
    'Volume': np.random.rand(n_samples) * 1000,
    'Close': np.random.uniform(100, 200, size=n_samples),
    'Signal': np.random.choice([-1, 0, 1], size=n_samples, p=[0.3, 0.4, 0.3])
})

X = df[['SMA_20', 'RSI', 'MACD', 'Volume']]
y = df['Signal']
X_train, X_test, y_train, y_test, close_train, close_test = train_test_split(X, y, df['Close'], test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

initial_cash = 10000
cash = initial_cash
position = 0
portfolio_values = []

for i in range(len(y_pred)):
    price = close_test.iloc[i]
    signal = y_pred[i]

    if signal == 1 and cash >= price:
        position = cash // price
        cash -= position * price
    elif signal == -1 and position > 0:
        cash += position * price
        position = 0

    portfolio_value = cash + (position * price)
    portfolio_values.append(portfolio_value)

returns = pd.Series(portfolio_values).pct_change().dropna()
total_return = (portfolio_values[-1] - initial_cash) / initial_cash
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

print("Final Portfolio Value:", round(portfolio_values[-1], 2))
print("Total Return: {:.2f}%".format(total_return * 100))
print("Sharpe Ratio:", round(sharpe_ratio, 2))

df = pd.read_csv("/content/ADANIPORTS (1).csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

close_prices = df['Close']
df['SMA_20'] = ta.trend.sma_indicator(close_prices, window=20)
df['RSI'] = ta.momentum.rsi(close_prices, window=14)
df['MACD'] = ta.trend.macd_diff(close_prices)

if 'Volume' in df.columns and 'Volume' in clf.feature_names_in_:
    features = ['SMA_20', 'RSI', 'MACD', 'Volume']
    X_data = df[features]
elif 'Volume' not in clf.feature_names_in_:
     features = ['SMA_20', 'RSI', 'MACD']
     X_data = df[features]
else:

    print("Warning: Mismatch in features used for training and available in data.")
    print("Features used for training:", clf.feature_names_in_)
    print("Features available in data:", df.columns)
    features = clf.feature_names_in_
    X_data = df[features]

df.dropna(subset=features, inplace=True)
X_data = df[features]

try:
    df['Predicted_Signal'] = clf.predict(X_data)
except NameError:
    print("Error: The classifier 'clf' was not found. Please ensure the cell training the RandomForestClassifier was run.")

    exit()


initial_cash = 10000
cash = initial_cash
position = 0
portfolio_values = []


for i, row in df.iterrows():
    price = row['Close']
    signal = row['Predicted_Signal']

    if signal == 1 and cash >= price:

        position = cash // price
        cash -= position * price

    elif signal == -1 and position > 0:

        cash += position * price
        position = 0


    portfolio_value = cash + (position * price)
    portfolio_values.append(portfolio_value)

df['Portfolio_Value'] = portfolio_values

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Portfolio_Value'], label='Portfolio Value', color='green')
plt.title('Portfolio Value Over Time (Based on Model Predictions)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

df['Portfolio_Return'] = df['Portfolio_Value'].pct_change()

returns = df['Portfolio_Return'].dropna()

total_return = (df['Portfolio_Value'].iloc[-1] - df['Portfolio_Value'].iloc[0]) / df['Portfolio_Value'].iloc[0] * 100

sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

print("Final Portfolio Value: ${:.2f}".format(df['Portfolio_Value'].iloc[-1]))
print("Total Return: {:.2f}%".format(total_return))
print("Sharpe Ratio: {:.2f}".format(sharpe_ratio))

!pip install newsapi-python
!pip install nltk

import pandas as pd

df = pd.read_csv("/content/ADANIPORTS (1).csv")

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values('Date').reset_index(drop=True)

print(df.columns)

!pip install ta

import pandas as pd
import numpy as np
import ta  # Technical Analysis library

df = pd.read_csv("/content/ADANIPORTS (1).csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
df['MACD'] = ta.trend.macd_diff(df['Close'])

bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['BB_Upper'] = bb.bollinger_hband()
df['BB_Lower'] = bb.bollinger_lband()

df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])


df['Chaikin'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20)

df.dropna(inplace=True)

df['Target_Regression'] = df['Close'].shift(-1)

threshold = 0.01
df['Return_pct'] = df['Close'].pct_change().shift(-1)

df['Signal'] = 0
df.loc[df['Return_pct'] > threshold, 'Signal'] = 1
df.loc[df['Return_pct'] < -threshold, 'Signal'] = -1

df.dropna(inplace=True)

print(df[['Date', 'Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'OBV', 'Chaikin',
          'BB_Upper', 'BB_Lower', 'Target_Regression', 'Signal']].head())

df['Target_Regression'] = df['Close'].shift(-1)
df.dropna(inplace=True)
print(df[['Date', 'Close', 'Target_Regression']].head())

threshold = 0.01

df['Return_pct'] = df['Close'].pct_change().shift(-1)
df['Signal'] = 0

df.loc[df['Return_pct'] > threshold, 'Signal'] = 1
df.loc[df['Return_pct'] < -threshold, 'Signal'] = -1

df.dropna(inplace=True)
print(df[['Date', 'Close', 'Signal']].head())

print(df[['Date', 'Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD',
          'BB_Upper', 'BB_Lower', 'OBV', 'Chaikin', 'Target_Regression', 'Signal']].head())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

feature_cols = [
    'SMA_20', 'EMA_20', 'RSI', 'MACD', 'OBV', 'Chaikin',
    'BB_Upper', 'BB_Lower',
    'lag_1', 'lag_2', 'lag_3', 'RSI_lag1', 'MACD_lag1',
    'rolling_mean_3', 'rolling_mean_7', 'rolling_std_7',
    'rolling_max_7', 'rolling_min_7', 'momentum_7'
]

X = df[feature_cols]
y = df['Signal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

df['lag_1'] = df['Close'].shift(1)
df['lag_2'] = df['Close'].shift(2)
df['lag_3'] = df['Close'].shift(3)
df['lag_4'] = df['Close'].shift(4)
df['lag_5'] = df['Close'].shift(5)
df['RSI_lag1'] = df['RSI'].shift(1)
df['MACD_lag1'] = df['MACD'].shift(1)

df['rolling_mean_3'] = df['Close'].rolling(window=3).mean()
df['rolling_mean_7'] = df['Close'].rolling(window=7).mean()
df['rolling_std_7'] = df['Close'].rolling(window=7).std()  # Volatility
df['rolling_max_7'] = df['Close'].rolling(window=7).max()
df['rolling_min_7'] = df['Close'].rolling(window=7).min()

df['momentum_7'] = df['Close'] - df['rolling_mean_7']

df.dropna(inplace=True)

print(df[['Date', 'Close', 'lag_1', 'rolling_mean_3', 'momentum_7']].head())

import matplotlib.pyplot as plt
import seaborn as sns

importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='coolwarm')
plt.title("Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

coefs = np.abs(log_model.coef_[0])

log_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': coefs
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=log_importance, x='Importance', y='Feature', palette='crest')
plt.title("Logistic Regression – Feature Importance (abs(coef))")
plt.tight_layout()
plt.show()

import xgboost as xgb
from sklearn.model_selection import train_test_split

label_mapping = {-1: 0, 0: 1, 1: 2}
y_train_encoded = y_train.map(label_mapping)
y_test_encoded = y_test.map(label_mapping)

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

xgb_model.fit(X_train, y_train_encoded)

xgb.plot_importance(xgb_model, importance_type='gain', title='XGBoost Feature Importance', height=0.5)
plt.show()

df = pd.read_csv("/content/ADANIPORTS (1).csv")

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
df['MACD'] = ta.trend.macd_diff(df['Close'])

bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['BB_Upper'] = bb.bollinger_hband()
df['BB_Lower'] = bb.bollinger_lband()

df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

df['Chaikin'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20)

df['lag_1'] = df['Close'].shift(1)
df['lag_2'] = df['Close'].shift(2)
df['lag_3'] = df['Close'].shift(3)
df['RSI_lag1'] = df['RSI'].shift(1)
df['MACD_lag1'] = df['MACD'].shift(1)

df['rolling_mean_3'] = df['Close'].rolling(window=3).mean()
df['rolling_mean_7'] = df['Close'].rolling(window=7).mean()
df['rolling_std_7'] = df['Close'].rolling(window=7).std()
df['rolling_max_7'] = df['Close'].rolling(window=7).max()
df['rolling_min_7'] = df['Close'].rolling(window=7).min()
df['momentum_7'] = df['Close'] - df['rolling_mean_7']


threshold = 0.01

df['Return_pct'] = df['Close'].pct_change().shift(-1)
df['Signal'] = 0

df.loc[df['Return_pct'] > threshold, 'Signal'] = 1
df.loc[df['Return_pct'] < -threshold, 'Signal'] = -1

df.dropna(inplace=True)

label_mapping = {-1: 0, 0: 1, 1: 2}
df['Signal_encoded'] = df['Signal'].map(label_mapping)

feature_cols = [
    'SMA_20', 'EMA_20', 'RSI', 'MACD', 'OBV', 'Chaikin',
    'BB_Upper', 'BB_Lower',
    'lag_1', 'lag_2', 'lag_3', 'RSI_lag1', 'MACD_lag1',
    'rolling_mean_3', 'rolling_mean_7', 'rolling_std_7',
    'rolling_max_7', 'rolling_min_7', 'momentum_7'
]

missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    print(f"Warning: Missing features in DataFrame after processing: {missing_cols}")


X = df[feature_cols]
y = df['Signal_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

y_pred_encoded = xgb_model.predict(X_test)

reverse_label_mapping = {v: k for k, v in label_mapping.items()}
y_test_original = pd.Series(y_test.values).map(reverse_label_mapping)
y_pred = pd.Series(y_pred_encoded).map(reverse_label_mapping)


print("\n XGBoost Model Evaluation:")
print("Accuracy:", accuracy_score(y_test_original, y_pred))
print("Classification Report:\n", classification_report(y_test_original, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_original, y_pred, labels=[-1, 0, 1]))

xgb.plot_importance(xgb_model, importance_type='gain', title='XGBoost Feature Importance', height=0.5, max_num_features=10)
plt.show()

importances = xgb_model.feature_importances_
feature_importance_df_xgb = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

num_features_to_plot = 20
feature_importance_df_xgb_plot = feature_importance_df_xgb.head(num_features_to_plot)
plot_height = max(4, len(feature_importance_df_xgb_plot) * 0.4)

plt.figure(figsize=(10, plot_height))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df_xgb_plot, palette='viridis')
plt.title("XGBoost Feature Importances (Gain)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

correlation_features = [
    'SMA_20', 'EMA_20', 'RSI', 'MACD', 'OBV', 'Chaikin',
    'BB_Upper', 'BB_Lower',
    'lag_1', 'lag_2', 'lag_3',
    'RSI_lag1', 'MACD_lag1',
    'rolling_mean_3', 'rolling_mean_7',
    'rolling_std_7', 'rolling_max_7',
    'rolling_min_7', 'momentum_7',
    'Signal'
]

corr_matrix = df[correlation_features].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Features")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

missing = [col for col in correlation_features if col not in df.columns]
print("Missing columns from your CSV:", missing)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
    df['RSI'] = calculate_rsi(df['Close'])

plt.figure(figsize=(12, 4))
plt.plot(df['RSI'], color='purple', label='RSI')
plt.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
plt.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
plt.title("Relative Strength Index (RSI)")
plt.ylabel("RSI Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

short_ema = df['Close'].ewm(span=12, adjust=False).mean()
long_ema = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = short_ema - long_ema

plt.figure(figsize=(12, 4))
plt.plot(df['MACD'], color='brown', label='MACD Line')
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.title("MACD Line")
plt.ylabel("MACD Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load and preprocess the data
df = pd.read_csv("/content/ADANIPORTS (1).csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Create technical and lag features
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['EMA_20'] = df['Close'].ewm(span=20).mean()
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)
df['Target'] = df['Close'].shift(-1)

# Drop rows with missing values
df.dropna(inplace=True)

# -------------------------------
# 📅 Select random year (2023 here)
# -------------------------------
target_year = 2023
train_df = df[df.index.year < target_year]
test_df = df[df.index.year == target_year]

features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'EMA_20', 'Lag_1', 'Lag_2', 'Lag_3']

X_train = train_df[features]
y_train = train_df['Target']
X_test = test_df[features]
y_test = test_df['Target']

# -------------------------------
# 🧠 Train Linear Regression Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# -------------------------------
# 📊 Visualization
# -------------------------------
plt.figure(figsize=(14, 6))
plt.plot(test_df.index, y_test, label='Actual Close Price', linewidth=2)
plt.plot(test_df.index, y_pred, label='Predicted Close Price', linestyle='--')
plt.title(f"📈 Close Price Prediction for {target_year} using Linear Regression")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Show prediction table
result_df = pd.DataFrame({
    'Date': test_df.index,
    'Actual': y_test.values,
    'Predicted': y_pred
})
print(result_df.head(10))

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random

df = pd.read_csv("/content/ADANIPORTS (1).csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['EMA_20'] = df['Close'].ewm(span=20).mean()
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)
df['Target'] = df['Close'].shift(-1)

df.dropna(inplace=True)

available_years = df.index.year.unique().tolist()
print("Available years after preprocessing:", available_years)

target_year = random.choice(available_years)
print(f"\n Randomly selected prediction year: {target_year}")


train_df = df[df.index.year < target_year]
test_df = df[df.index.year == target_year]

if test_df.empty or train_df.empty:
    print(f"Not enough data for year {target_year}. Try running again.")
else:
    features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'EMA_20', 'Lag_1', 'Lag_2', 'Lag_3']
    X_train = train_df[features]
    y_train = train_df['Target']
    X_test = test_df[features]
    y_test = test_df['Target']

    model = LinearRegression()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    plt.figure(figsize=(14, 6))
    plt.plot(test_df.index, y_test, label='Actual Close Price', linewidth=2)
    plt.plot(test_df.index, y_pred, label='Predicted Close Price', linestyle='--')
    plt.title(f" Close Price Prediction for {target_year} using Linear Regression")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    result_df = pd.DataFrame({
        'Date': test_df.index,
        'Actual': y_test.values,
        'Predicted': y_pred
    })
    print("\n Sample Predictions:")
    print(result_df.head(10))