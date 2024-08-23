import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import DataLoader
import os

# Created graph for GCN Path
GRAPH_PATH = "./4_edge_cosSim_AP2D_count_nonportal_discon.pt"
# Model save path
MODEL_PATH = ""

# Best model parameters
final_params = {
    'hidden_layers': [17],
    'activation_function': F.leaky_relu,
    'dropout': True,
    'dropout_rate': 0.15,
    'batch_norm': True,
    'batch_size': 150,
    'learning_rate': 35,
    'epochs': 4000
}

# Define the model
class SAGENet(nn.Module):
    def __init__(self, in_feats, hidden_layers, activation, dropout, dropout_rate, batch_norm):
        super(SAGENet, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if batch_norm else None
        self.activation = activation
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

        # Input layer
        self.layers.append(SAGEConv(in_feats, hidden_layers[0]))
        if self.batch_norm:
            self.bns.append(nn.BatchNorm1d(hidden_layers[0]))

        # Hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(SAGEConv(hidden_layers[i-1], hidden_layers[i]))
            if self.batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden_layers[i]))

        # Output layer
        self.layers.append(SAGEConv(hidden_layers[-1], 1))  # Output dimension is 1 for regression

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, edge_index)
            if self.batch_norm:
                x = self.bns[i](x)
            x = self.activation(x)
            if self.dropout:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.layers[-1](x, edge_index)
        return x

# Train the model
def train(data, model, device, epochs, batch_size, lr, patience=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0

    # DataLoader for batching
    loader = DataLoader([data], batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch).squeeze()

            # Compute loss
            loss = F.mse_loss(logits[batch.train_mask], batch.y[batch.train_mask].float())
            loss.backward()
            optimizer.step()

            # Evaluation
            with torch.no_grad():
                val_loss = F.mse_loss(logits[batch.val_mask], batch.y[batch.val_mask].float())

            # Save the best validation loss
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                best_model = model
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: train loss = {loss:.3f}, val loss = {val_loss:.3f} (best = {best_val_loss:.3f})")

        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch} epochs with best validation loss: {best_val_loss:.3f}")
            break

    return best_model

# Save model
def save_model(model, file_path, params):
    state = {
        'model_state_dict': model.state_dict(),
        'params': params
    }
    torch.save(state, file_path)

# Main script
if __name__ == "__main__":
    # Load data
    data = torch.load(GRAPH_PATH)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and train the model
    model = SAGENet(data.num_features, final_params['hidden_layers'], final_params['activation_function'], 
                    final_params['dropout'], final_params['dropout_rate'], final_params['batch_norm']).to(device)
    model = train(data, model, device, final_params['epochs'], final_params['batch_size'], 
                  final_params['learning_rate'], patience=150)

    # Save the final model
    save_model(model, MODEL_PATH, final_params)
