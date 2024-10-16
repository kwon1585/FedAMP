import os
import torch
import torch.optim as optim
import numpy as np
from model import SimpleCNN
from data import load_dataset, create_dataloaders, load_client_datasets
from train import local_update, evaluate
from utils.fedamp import aggregate_models_amp, aggregate_models_avg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_clients = 62
num_iterations = 15
alpha = 0.1
lambda_param = 0.5
sigma = 1.0
batch_size = 32
test_size_per_client = 100

# Load datasets
train_dataset, test_dataset = load_dataset()

# Create dataloaders
if os.path.exists("./data/client_datasets.pkl"):
    client_dataloaders, client_test_dataloaders = load_client_datasets(train_dataset, batch_size=batch_size)
else : client_dataloaders, client_test_dataloaders = create_dataloaders(train_dataset, num_clients, test_size_per_client, batch_size)

# Initialize models, optimizers, and loss function
clients_models_amp = [SimpleCNN().to(device) for _ in range(num_clients)]
clients_models_avg = [SimpleCNN().to(device) for _ in range(num_clients)]
clients_optimizers_amp = [optim.SGD(model.parameters(), lr=0.01) for model in clients_models_amp]
clients_optimizers_avg = [optim.SGD(model.parameters(), lr=0.01) for model in clients_models_avg]
clients_criterion = torch.nn.CrossEntropyLoss()

cloud_models_amp = [SimpleCNN().to(device) for _ in range(num_clients)]
cloud_model_avg = SimpleCNN().to(device)

all_accuracies_amp = {'mean': [], 'max': [], 'min': []}
all_accuracies_avg = {'mean': [], 'max': [], 'min': []}

# FedAMP training
for iteration in range(num_iterations):
    for i in range(num_clients):
        local_update(clients_models_amp[i], cloud_models_amp[i], clients_optimizers_amp[i], clients_criterion, client_dataloaders[i], alpha, lambda_param, device)

    # Evaluate FedAMP
    accuracies_amp = [evaluate(clients_models_amp[i], client_test_dataloaders[i], clients_criterion, device=device)[1] for i in range(num_clients)]
    all_accuracies_amp['mean'].append(np.mean(accuracies_amp))
    all_accuracies_amp['max'].append(np.max(accuracies_amp))
    all_accuracies_amp['min'].append(np.min(accuracies_amp))
    
    # Cloud Update (FedAMP)
    amp_model_state = aggregate_models_amp(clients_models_amp, alpha, sigma, num_clients)
    for i in range(num_clients):
        cloud_models_amp[i].load_state_dict(amp_model_state[i])

# FedAvg training
for iteration in range(num_iterations):
    for i in range(num_clients):
        local_update(clients_models_avg[i], None, clients_optimizers_avg[i], clients_criterion, client_dataloaders[i], device=device)

    accuracies_avg = [evaluate(clients_models_avg[i], client_test_dataloaders[i], clients_criterion, device=device)[1] for i in range(num_clients)]
    all_accuracies_avg['mean'].append(np.mean(accuracies_avg))
    all_accuracies_avg['max'].append(np.max(accuracies_avg))
    all_accuracies_avg['min'].append(np.min(accuracies_avg))


# Save results
import pickle
results_dir = './results'
fedamp_results_path = f'{results_dir}/fedamp_accuracies.pkl'
fedavg_results_path = f'{results_dir}/fedavg_accuracies.pkl'

with open(fedamp_results_path, 'wb') as f:
    pickle.dump(all_accuracies_amp, f)

with open(fedavg_results_path, 'wb') as f:
    pickle.dump(all_accuracies_avg, f)