import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import DataLoader
import torch.nn.init as init
import matplotlib.pyplot as plt
import os
import itertools

# Load the graph and data
data = torch.load('4_edge_cosSim_AP2D_count_nonportal_discon.pt')

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

def weights_init_normal(m):
    if isinstance(m, SAGEConv):
        init.normal_(m.lin_l.weight.data, mean=0.0, std=0.02)
        if m.lin_l.bias is not None:
            init.constant_(m.lin_l.bias.data, 0.0)
        init.normal_(m.lin_r.weight.data, mean=0.0, std=0.02)
        if m.lin_r.bias is not None:
            init.constant_(m.lin_r.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight.data, mean=0.0, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

class SAGENet(nn.Module):
    def __init__(self, in_feats, hidden_layers, activation=F.relu, dropout=False, dropout_rate=0.5, batch_norm=False):
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

def train(data, model, device, epochs, batch_size, lr, patience=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr= lr)
    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0

    # Create DataLoader for batching
    loader = DataLoader([data], batch_size=batch_size, shuffle=True, num_workers=0)

    train_losses = []
    val_losses = []

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

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())

            # Save the best validation loss
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                best_model = model
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if epoch % 5 == 0:
            print(
                f"In epoch {epoch}, train loss: {loss:.3f}, val loss: {val_loss:.3f} (best {best_val_loss:.3f})"
            )

        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch} epochs with best validation loss: {best_val_loss:.3f}")
            break

    actual_epochs = len(train_losses)  # Track the actual number of epochs run
    return best_model, train_losses, val_losses, best_val_loss, actual_epochs

def create_train_plot(epochs, train_losses, val_losses, learning_rate, hidden_layers, batch_size, plot_path):
    # Plot training and validation losses
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')

    # Add subtitle with parameters
    subtitle_text = f'Parameters: LR={learning_rate}, HL={len(hidden_layers)}, HLS={hidden_layers}, BS={batch_size}\n'
    plt.text(0.5, 1.05, subtitle_text, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

def save_model(model, file_path, params):
    state = {
        'model_state_dict': model.state_dict(),
        'params': params
    }
    torch.save(state, file_path)

def get_activation_function_name(activation_function):
    if activation_function == F.relu:
        return 'relu'
    elif activation_function == F.leaky_relu:
        return 'leakyrelu'
    else:
        return 'unknown'
    

# Function to perform grid search for hyperparameters and train models
def grid_search(data, device, param_grid):

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []

    for params in param_combinations:
        hidden_layers = params['hidden_layers']
        activation_function = params.get('activation_function', F.relu)
        dropout = params.get('dropout', False)
        dropout_rate = params.get('dropout_rate', 0.5)
        batch_norm = params.get('batch_norm', False)
        batch_size = params.get('batch_size', 32)
        learning_rate = params.get('learning_rate', 0.01)
        epochs = params.get('epochs', 1000)

        act_name = get_activation_function_name(activation_function)

        param_set = {
            'hidden_layers': hidden_layers,
            'activation_function': activation_function,
            'dropout': dropout,
            'dropout_rate': dropout_rate, 
            'batch_norm': batch_norm,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        
        model = SAGENet(data.num_features, hidden_layers, activation_function, dropout, dropout_rate, batch_norm).to(device)
        # # Apply the weight initialization
        # model.apply(weights_init_normal)
        model, train_losses, val_losses, best_val_loss, act_epochs = train(data, model, device, epochs, batch_size, learning_rate, patience=150)

        # Save the model
        save_dir = 'gridSearch_models'
        model_path = os.path.join(save_dir,
            f"model_hl{hidden_layers}_dr{dropout}_drR{dropout_rate}_bn{batch_norm}_bs{batch_size}_{act_name}_lr{learning_rate}.pt"
        )
        #torch.save(model.state_dict(), model_path)
        save_model(model, model_path, param_set)

        # Save the training plot
        plot_path = os.path.join(save_dir,
            f"plot_hl{hidden_layers}_dr{dropout}_drR{dropout_rate}_bn{batch_norm}_bs{batch_size}_{act_name}_lr{learning_rate}.png"
        )
        create_train_plot(act_epochs, train_losses, val_losses, learning_rate, hidden_layers, batch_size, plot_path)

# Define hyperparameter grid
param_grid = {
    'hidden_layers': [[17], [16], [18]], #[1000], [500], [100], [1000,500], [1000,500,100], [32], [32,16], [32,16,32], [16,32,16]
    'activation_function': [F.leaky_relu], #F.relu, 
    'dropout': [True], # True, 
    'dropout_rate': [.11,.13,.15, .17, .19], #, .4, 0.5, .6
    'batch_norm': [True],
    'batch_size': [130,135,140,145,150,155,160,165,170],
    'learning_rate': [.31,33,35,.37,.39],
    'epochs': [4000]
}

# Run grid search
grid_search(data, device, param_grid)
