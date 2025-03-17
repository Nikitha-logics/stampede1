import torch
import torch.nn as nn
import numpy as np

class LSTMRiskPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super(LSTMRiskPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last time step
        out = self.sigmoid(out)
        return out

class RiskPredictor:
    def __init__(self, model_path="../models/lstm_model.pth", sequence_length=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMRiskPredictor().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.sequence_length = sequence_length
        self.history = []  # Fixed line: Removed the extra 'm'

    def predict(self, density, sudden_movement):
        # Add new data to history
        self.history.append([density, int(sudden_movement)])
        if len(self.history) > self.sequence_length:
            self.history.pop(0)
        
        # Prepare input for LSTM
        if len(self.history) < self.sequence_length:
            return 0.0  # Not enough data for prediction
        
        input_data = np.array(self.history, dtype=np.float32)
        input_tensor = torch.tensor(input_data).unsqueeze(0).to(self.device)  # [1, seq_len, 2]
        
        # Predict risk
        with torch.no_grad():
            risk_score = self.model(input_tensor).item()
        return risk_score

if __name__ == "__main__":
    # Simulated test
    predictor = RiskPredictor()
    for i in range(15):
        density = np.random.uniform(0, 1)  # Simulated density
        sudden_movement = np.random.choice([True, False])  # Simulated sudden movement
        risk = predictor.predict(density, sudden_movement)
        print(f"Frame {i+1}: Density={density:.2f}, Sudden Movement={sudden_movement}, Risk={risk:.2f}")