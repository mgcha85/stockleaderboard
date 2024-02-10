import pandas as pd
from PyEMD import EMD
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from database import Database


class ELMRegressor:
    def __init__(self, n_hidden_units):
        self.n_hidden_units = n_hidden_units
        self.alpha = 1.0  # Regularization strength for Ridge Regression

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        # Randomly initialize weights and biases
        self.input_weights = np.random.normal(size=[X.shape[1], self.n_hidden_units])
        self.biases = np.random.normal(size=[self.n_hidden_units])

        # Calculate hidden layer output
        H = self._sigmoid(np.dot(X, self.input_weights) + self.biases)

        # Train output weights using Ridge Regression
        self.output_weights = Ridge(alpha=self.alpha, fit_intercept=False)
        self.output_weights.fit(H, y)

    def predict(self, X):
        H = self._sigmoid(np.dot(X, self.input_weights) + self.biases)
        return self.output_weights.predict(H)

# Assuming 'db' is an instance of 'Database' and 'df_list' is the DataFrame containing the trading log
def prepare_data_for_elm(df_list, symbol, db, sequence_length=400, feature_length=100):
    all_features = []
    all_labels = []

    for idx, row in df_list.iterrows():
        entry_date = row['Entry Date']
        profit = row['Abs Profit']
        label = 1 if profit > 0 else 0

        query = f"SELECT * FROM `{symbol}_1m` WHERE time <= '{entry_date}' ORDER BY time DESC LIMIT {sequence_length}"
        df = pd.read_sql(query, db.conn, index_col='time')
        if df.shape[0] < sequence_length:
            continue

        emd = EMD()
        imfs = emd(df['close'].values)

        # Ensure consistent feature length for each IMF
        imf_features = np.array([np.pad(imf, (0, max(0, feature_length - len(imf))), 'constant')[:feature_length] for imf in imfs])

        # Flatten the array and ensure it's of consistent size
        flat_features = imf_features.flatten()
        if len(flat_features) < feature_length * len(imfs):
            flat_features = np.pad(flat_features, (0, feature_length * len(imfs) - len(flat_features)), 'constant')

        print(flat_features.shape)
        all_features.append(flat_features)
        all_labels.append(label)

    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.float32)


# Prepare the data
db = Database()
algo = 'AlligatorStrategy'
symbol = "BTCUSDT"

df_list = pd.read_excel(f'{algo}.xlsx')

features, labels = prepare_data_for_elm(df_list, symbol, db)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# Initialize and train the ELM model
elm_model = ELMRegressor(n_hidden_units=100)
elm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = elm_model.predict(X_test)
accuracy = np.mean((y_pred > 0.5) == y_test)
print(f'ELM model accuracy: {accuracy:.2f}')
