import pandas as pd
import sqlite3
from database import Database
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from database import Database
from sklearn.decomposition import PCA
from tqdm import tqdm
import pickle as pkl


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.bias = bias
        self.attention = nn.Linear(feature_dim, 1, bias=bias)

    def forward(self, x):
        # Check the dimension of x
        if x.dim() == 3:
            eij = self.attention(x).squeeze(2)
        elif x.dim() == 2:
            eij = self.attention(x).squeeze(1)  # Assuming x is [batch, features]
        else:
            raise ValueError("Unexpected input dimensions in Attention layer")

        eij = torch.tanh(eij)
        a = torch.softmax(eij, dim=1)
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class LSTMPredictorWithAttention(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(LSTMPredictorWithAttention, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=0.5,
            batch_first=True
        )
        self.attention = Attention(n_hidden, seq_len)
        self.linear = nn.Linear(n_hidden, 2)

    def forward(self, sequences):
        lstm_out, _ = self.lstm(sequences)
        attn_out = self.attention(lstm_out)
        y_pred = self.linear(attn_out)
        return y_pred
    
# Define the DNN model
class DNNPredictor(nn.Module):
    def __init__(self, n_input, n_hidden_layers, n_output):
        super(DNNPredictor, self).__init__()
        layers = []
        for n_hidden in n_hidden_layers:
            layers.append(nn.Linear(n_input, n_hidden))
            layers.append(nn.ReLU())
            n_input = n_hidden
        layers.append(nn.Linear(n_hidden_layers[-1], n_output))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Function to apply PCA
def apply_pca(data_tensor, n_components=None):
    pca = PCA(n_components=n_components)
    # Assuming the data_tensor is of shape [samples, seq_len, features]
    # We transpose it to [samples, features, seq_len] and then reshape to [samples, features * seq_len]
    # so that each sequence is treated as a single sample with multiple features.
    samples, seq_len, features = data_tensor.size()
    flattened_data = data_tensor.transpose(1, 2).reshape(samples, features * seq_len)
    
    # If n_components is None, PCA will set it to min(n_samples, n_features)
    pca_data = pca.fit_transform(flattened_data.numpy())
    pca_tensor = torch.tensor(pca_data, dtype=torch.float32)
    return pca_tensor, pca.explained_variance_ratio_

# You may want to dynamically set n_components based on the explained variance ratio
# For example, you want to keep 95% of variance explained
def find_n_components_for_variance(data_tensor, variance_threshold=0.95):
    _, seq_len, features = data_tensor.size()
    pca = PCA()
    flattened_data = data_tensor.view(-1, seq_len * features).numpy()
    pca.fit(flattened_data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.where(cumulative_variance >= variance_threshold)[0][0] + 1
    return n_components

def calculate_ema(df, periods, tag="1m"):
    for period in periods:
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return df

# Define the LSTM model
class LSTMPredictor(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(LSTMPredictor, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=0.5
        )
        
        self.linear = nn.Linear(in_features=n_hidden, out_features=2)

    def forward(self, sequences):
        lstm_out, _ = self.lstm(sequences.view(len(sequences), self.seq_len, -1))
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred


# Function to resample and calculate EMAs for different timeframes
def resample_and_calculate_ema(df, timeframe, periods, tag='1m'):
    resampled_df = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df = calculate_ema(resampled_df, periods, tag)
    df.columns = [x + '_' + tag for x in df.columns]
    return df

# Assuming 'db' is an instance of 'Database' and 'df_list' is the DataFrame containing the trading log
def create_datasets(df_list, sequence_length=400):
    all_data = []
    all_labels = []
    scaler = StandardScaler()

    for idx, row in tqdm(df_list.iterrows()):
        entry_date = row['Entry Date']
        profit = row['Abs Profit']
        label = 1 if profit > 0 else 0  # 1 for profit, 0 for loss

        query = f"SELECT * FROM `BTCUSDT_1m` WHERE time <= '{entry_date}' ORDER BY time DESC LIMIT {sequence_length * 60}"
        df = pd.read_sql(query, db.engine, index_col='time')
        if df.shape[0] < sequence_length:
            continue
        df = df.sort_index()

        query = f"SELECT * FROM `{symbol}_1d` WHERE time <= '{entry_date}' ORDER BY time DESC LIMIT {sequence_length}"
        df_d = pd.read_sql(query, db.engine, index_col='time')
        if df_d.shape[0] < sequence_length:
            continue
        df_d = df_d.sort_index()
        df_d.columns = [x + '_1d' for x in df_d.columns]

        ema_periods = [5, 20, 60, 200]
        df_1m = calculate_ema(df, ema_periods)[-sequence_length:]
        df_5m = resample_and_calculate_ema(df, '5T', ema_periods, '5m')[-sequence_length:]
        df_15m = resample_and_calculate_ema(df, '15T', ema_periods, '15m')[-sequence_length:]
        df_1h = resample_and_calculate_ema(df, '1H', ema_periods, '1h')[-sequence_length:]
        df_1m.columns = [x + '_1m' for x in df_1m.columns]
        combined_df = np.hstack([df_1m.values, df_5m.values, df_15m.values, df_1h.values, df_d.values])

        # Normalize the data
        scaled_data = scaler.fit_transform(combined_df[int(sequence_length/2):, :])
        all_data.append(scaled_data)
        all_labels.append(label)

    all_data = np.array(all_data, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)

    # Convert to PyTorch tensors
    data_tensor = torch.tensor(all_data)
    labels_tensor = torch.tensor(all_labels).long()

    return data_tensor, labels_tensor


if __name__ == '__main__':
    db = Database()
    algo = 'DualBollingerBandStrategy_Trades'
    symbol = "BTCUSDT"
    sequence_length = 400
    n_hidden = 50
    n_output = 2
    n_epochs = 200

    df_list = pd.read_excel(f'{algo}.xlsx')

    # Prepare the data
    data_tensor, labels_tensor = create_datasets(df_list, sequence_length)
    with open(f"{algo}.pkl", "wb") as f:
        pkl.dump([data_tensor, labels_tensor], f)

    dataset = TensorDataset(data_tensor, labels_tensor)

    # Split dataset into training and validation datasets
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

    # Initialize the model
    n_input = data_tensor.size(1)  # Assuming the data_tensor's shape is [samples, features, seq_length]
    n_features = data_tensor.size(2)  # Number of features
    dnn_model = LSTMPredictorWithAttention(n_features, n_hidden, int(sequence_length/2), n_output)
    dnn_model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dnn_model.parameters(), lr=0.001)

    # Training Loop
    for epoch in range(n_epochs):
        dnn_model.train()  # Set the model to training mode
        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)  # Move data to device
            optimizer.zero_grad()
            y_pred = dnn_model(batch)
            loss = loss_function(y_pred, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch} loss: {loss.item()}')

    # Evaluation Loop
    dnn_model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        correct, total = 0, 0
        for batch, labels in val_loader:
            batch, labels = batch.to(device), labels.to(device)  # Move data to device
            y_pred = dnn_model(batch)
            predicted_labels = torch.argmax(y_pred, 1)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

    print(f'Validation accuracy with DNN: {100 * correct / total:.2f}%')
