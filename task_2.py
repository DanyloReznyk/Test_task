import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from constants import general_location

# Extracted folder path
data_path = 'task_2/data'

def read_csv_folder(folder_path, file_symbol):
    numbers_list = []

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(f"{file_symbol}.csv"):
            file_path = os.path.join(folder_path, filename)

            # Read numbers from the CSV file
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    number = float(row[0])
                    numbers_list.append(number)

    return numbers_list

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

def validation_save(autoencoder, i):
    validation_path = os.path.join(general_location, 'validation')
    data_path = os.path.join(general_location, 'data')

    if not os.path.exists(validation_path):
        os.mkdir(validation_path)

    for filename in os.listdir(data_path):
        if filename.endswith(f"{i}.csv"):

            file_path = os.path.join(data_path, filename)
            data_file = pd.read_csv(file_path)

            points_prediction = autoencoder(torch.tensor(data_file.values, dtype=torch.float32))
            output = pd.DataFrame(points_prediction.detach().numpy(), columns=['value'])

            output.to_csv(os.path.join(validation_path, filename), index=False)

# I decided to do validation for each curve noise level separately. this approach aims to enhance the overall data quality by addressing specific types of noise that may affect different curves differently.
# I also decided to conduct training separately for each class of noise.
# If the dataset did not have a distribution of noise ranks, then it would be worthwhile to make an additional network that would first distribute them by noise classes.
for i in range(1, 10):
    X = read_csv_folder(data_path, i)
    y = read_csv_folder(data_path, '0')
    data = pd.DataFrame({'features': X, 'target': y})

    X_train, X_val, y_train, y_val = train_test_split(data['features'], data['target'], test_size=0.2, random_state=2023)

    X_train = np.array(X_train).reshape(-1, 1)
    X_val = np.array(X_val).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    y_val = np.array(y_val).reshape(-1, 1)

    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    y_train_scaled = scaler.fit_transform(y_train)
    y_val_scaled = scaler.transform(y_val)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
    
    autoencoder = Autoencoder()

    #I decided use Reconstruction Error(MSE) because it measures the difference between the input data and the output reconstructed by the autoencoder.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr = 0.001)

    early_stopping = EarlyStopping(tolerance = 5, min_delta = 0.0005)
    train_losses = []
    val_losses = [] 
    for epoch in range(100): 
        optimizer.zero_grad()
        outputs = autoencoder(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Print training statistics
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

        train_losses.append(loss.item())

        # Validation
        with torch.no_grad():
            val_outputs = autoencoder(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            print(f'Validation Loss: {val_loss.item():.4f}')

            val_losses.append(val_loss.item())

        # Check early stopping
        early_stopping(train_losses[-1], val_losses[-1])
        if early_stopping.early_stop:
            print("We stop at epoch:", epoch)
            break

    validation_save(autoencoder, i)

    plt.plot(range(1, len(train_losses) + 1), train_losses, label = 'Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label = 'Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and validation loss For {i} level of noise')
    plt.show()

    torch.save(autoencoder.state_dict(), 'autoencoder_model_{i}.pth')
