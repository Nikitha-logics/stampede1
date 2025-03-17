import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from risk_prediction import LSTMRiskPredictor

# Generate a simple synthetic dataset
num_sequences = 1000
sequence_length = 10
data = []
labels = []

for _ in range(num_sequences):
    density = np.random.uniform(0, 1, sequence_length)
    sudden_movement = np.random.choice([0, 1], sequence_length)
    sequence = np.stack([density, sudden_movement], axis=1)
    label = 1 if np.mean(density) > 0.7 and np.sum(sudden_movement) > 5 else 0
    data.append(sequence)
    labels.append(label)

data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.float32)

# Create DataLoader
dataset = TensorDataset(torch.tensor(data), torch.tensor(labels))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRiskPredictor(input_size=2, hidden_size=64, num_layers=2).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_data, batch_labels in dataloader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        outputs = model(batch_data).squeeze()
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

# Save the model
model_path = r"C:\Users\susan\OneDrive\Desktop\STAMPEDE\stampede_env\models\lstm_model.pth"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"lstm_model.pth saved to {model_path}")