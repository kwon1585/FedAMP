import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

num_clients = 62
num_iterations = 10
alpha = 0.1
lambda_param = 0.5
sigma = 1.0
batch_size = 32
test_size_per_client = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def attention_derivative(wi, wj, sigma):
    diff = np.linalg.norm(wi - wj)**2
    return np.exp(-diff / sigma)


def aggregate_models_amp(clients, alpha, sigma):
    new_models = []
    for i in range(num_clients):
        wi = np.concatenate([p.data.cpu().numpy().ravel() for p in clients[i].parameters()])
        weighted_sum = np.zeros_like(wi)
        total_weight = 0
        for j in range(num_clients):
            if i != j:
                wj = np.concatenate([p.data.cpu().numpy().ravel() for p in clients[j].parameters()])
                weight = attention_derivative(wi, wj, sigma)
                weighted_sum += weight * wj
                total_weight += weight
        new_model = (1 - alpha * total_weight) * wi + alpha * weighted_sum
        new_models.append(new_model)
        
    new_model_states = []
    for model_params in new_models:
        model_state = {}  
        start = 0
        for name, param in clients[0].named_parameters():
            num_params = param.numel()
            model_state[name] = torch.tensor(model_params[start:start + num_params]).reshape(param.shape)
            start += num_params
        new_model_states.append(model_state)
    return new_model_states

def aggregate_models_avg(clients):
    avg_model = clients[0].state_dict()
    for key in avg_model.keys():
        avg_model[key] = torch.stack([clients[i].state_dict()[key] for i in range(num_clients)], dim=0).mean(dim=0)
    return avg_model

def local_update(model, cloud_model, optimizer, criterion, dataloader, alpha=None, lambda_param=None):
    model.train()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if cloud_model and lambda_param:
            model_params = np.concatenate([p.data.cpu().numpy().ravel() for p in model.parameters()])
            cloud_params = np.concatenate([p.data.cpu().numpy().ravel() for p in cloud_model.parameters()])
            regularization = lambda_param * np.sum((model_params - cloud_params)**2)
            total_loss = loss + regularization
        else:
            total_loss = loss
        total_loss.backward()
        optimizer.step()

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("Start downloading dataset")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='./data', split='byclass', train=False, download=True, transform=transform)

def create_client_datasets(train_dataset, num_clients, test_size_per_client):
    clients_data = [[] for _ in range(num_clients)]
    digit_clients = list(range(10))
    upper_clients = list(range(10, 36))
    lower_clients = list(range(36, 62))
    
    for idx, (data, target) in enumerate(train_dataset):
        if target < 10:
            clients_data[np.random.choice(digit_clients)].append(idx)
        elif 10 <= target < 36:
            clients_data[np.random.choice(upper_clients)].append(idx)
        elif 36 <= target < 62:
            clients_data[np.random.choice(lower_clients)].append(idx)

    client_train_datasets = []
    client_test_datasets = []

    for indices in clients_data:
        train_indices, test_indices = random_split(indices, [len(indices) - test_size_per_client, test_size_per_client])
        client_train_datasets.append(Subset(train_dataset, train_indices))
        client_test_datasets.append(Subset(train_dataset, test_indices))

    return client_train_datasets, client_test_datasets

def assign_data_with_dominating_and_other_classes(train_dataset, clients_data, dominant_ratio=0.8):
    client_train_datasets = []
    client_test_datasets = []

    for indices in clients_data:
        dominant_size = int(len(indices) * dominant_ratio)
        other_size = len(indices) - dominant_size

        dominant_indices = indices[:dominant_size]
        other_indices = indices[dominant_size:]

        other_group_indices = []
        for i, idx in enumerate(other_indices):
            other_target = (train_dataset[idx][1] + np.random.randint(1, 62)) % 62  
            other_group_indices.append(other_target)

        train_indices, test_indices = random_split(indices, [len(indices) - test_size_per_client, test_size_per_client])
        client_train_datasets.append(Subset(train_dataset, train_indices))
        client_test_datasets.append(Subset(train_dataset, test_indices))

    return client_train_datasets, client_test_datasets


print("Create dataloaders")

client_train_datasets, client_test_datasets = create_client_datasets(train_dataset, num_clients, test_size_per_client)

client_dataloaders = [DataLoader(client_train_datasets[i], batch_size=batch_size, shuffle=True) for i in range(num_clients)]
client_test_dataloaders = [DataLoader(client_test_datasets[i], batch_size=batch_size, shuffle=False) for i in range(num_clients)]

print("Client models initialized")

clients_models_amp = [SimpleCNN().to(device) for _ in range(num_clients)]
clients_models_avg = [SimpleCNN().to(device) for _ in range(num_clients)]
clients_optimizers_amp = [optim.SGD(model.parameters(), lr=0.01) for model in clients_models_amp]
clients_optimizers_avg = [optim.SGD(model.parameters(), lr=0.01) for model in clients_models_avg]
clients_criterion = nn.CrossEntropyLoss()

print("Cloud models initialized")

cloud_models_amp = [SimpleCNN().to(device) for _ in range(num_clients)]
cloud_model_avg = SimpleCNN().to(device)

def evaluate(model, dataloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(dataloader.dataset)
    accuracy = 100. * correct / len(dataloader.dataset)
    return test_loss, accuracy

all_accuracies_amp = []
all_accuracies_avg = []

# FedAMP 학습
print("Start FedAMP training")

for iteration in range(num_iterations):
    print("Iteration", iteration)
    
    # Local Update (FedAMP)
    for i in range(num_clients):
        print(i, end=' ')
        local_update(clients_models_amp[i], cloud_models_amp[i], clients_optimizers_amp[i], clients_criterion, client_dataloaders[i], alpha, lambda_param)
    print("")
    
    # 평가 (FedAMP)
    accuracies_amp = []
    for i in range(num_clients):
        print(i, end=' ')
        _, accuracy_amp = evaluate(clients_models_amp[i], client_test_dataloaders[i], clients_criterion)
        accuracies_amp.append(accuracy_amp)
    print("")
    
    avg_accuracy_amp = sum(accuracies_amp) / num_clients
    all_accuracies_amp.append(avg_accuracy_amp)
    
    print(f"Iteration {iteration}: After Local Update (FedAMP) - Avg Test Accuracy = {avg_accuracy_amp:.2f}%")
    
    # Cloud Update (FedAMP)
    amp_model_state = aggregate_models_amp(clients_models_amp, alpha, sigma)
    for i in range(num_clients):
        print(i, end=' ')
        cloud_models_amp[i].load_state_dict(amp_model_state[i])
    print("")
    
    # # 평가 (FedAMP) - Cloud Update 후
    # accuracies_amp = []
    # for i in range(num_clients):
    #     print(i, end=' ')
    #     _, accuracy_amp = evaluate(clients_models_amp[i], client_test_dataloaders[i], clients_criterion)
    #     accuracies_amp.append(accuracy_amp)
    # print("")
    
    # avg_accuracy_amp = sum(accuracies_amp) / num_clients
    # all_accuracies_amp.append(avg_accuracy_amp)
    
    # print(f"Iteration {iteration}: After Cloud Update (FedAMP) - Avg Test Accuracy = {avg_accuracy_amp:.2f}%")

# FedAvg 학습
print("Start FedAvg training")

for iteration in range(num_iterations):
    print("Iteration", iteration)
    
    # Local Update (FedAvg)
    for i in range(num_clients):
        print(i, end=' ')
        local_update(clients_models_avg[i], None, clients_optimizers_avg[i], clients_criterion, client_dataloaders[i])
    print("")
    
    # 평가 (FedAvg)
    accuracies_avg = []
    for i in range(num_clients):
        print(i, end=' ')
        _, accuracy_avg = evaluate(clients_models_avg[i], client_test_dataloaders[i], clients_criterion)
        accuracies_avg.append(accuracy_avg)
    print("")
    
    avg_accuracy_avg = sum(accuracies_avg) / num_clients
    all_accuracies_avg.append(avg_accuracy_avg)
    
    print(f"Iteration {iteration}: After Local Update (FedAvg) - Avg Test Accuracy = {avg_accuracy_avg:.2f}%")
    
    # # Cloud Update (FedAvg)
    # avg_model_state = aggregate_models_avg(clients_models_avg)
    # for i in range(num_clients):
    #     print(i, end=' ')
    #     clients_models_avg[i].load_state_dict(avg_model_state)
    # print("")
    
    # # 평가 (FedAvg) - Cloud Update 후
    # accuracies_avg = []
    # for i in range(num_clients):
    #     print(i, end=' ')
    #     _, accuracy_avg = evaluate(clients_models_avg[i], client_test_dataloaders[i], clients_criterion)
    #     accuracies_avg.append(accuracy_avg)
    # print("")
    
    # avg_accuracy_avg = sum(accuracies_avg) / num_clients
    # all_accuracies_avg.append(avg_accuracy_avg)
    
    # print(f"Iteration {iteration}: After Cloud Update (FedAvg) - Avg Test Accuracy = {avg_accuracy_avg:.2f}%")


# print("Avg accuracy over all iterations (FedAMP vs FedAvg):")
# for idx in range(num_iterations):
#     print(f"Iteration {idx}, After Local Update: FedAMP = {all_accuracies_amp[2*idx]:.2f}%, FedAvg = {all_accuracies_avg[2*idx]:.2f}%")
#     print(f"Iteration {idx}, After Cloud Update: FedAMP = {all_accuracies_amp[2*idx+1]:.2f}%, FedAvg = {all_accuracies_avg[2*idx+1]:.2f}%")

import pickle

results_dir = './results'
fedamp_results_path = f'{results_dir}/fedamp_accuracies.pkl'
fedavg_results_path = f'{results_dir}/fedavg_accuracies.pkl'

with open(fedamp_results_path, 'wb') as f:
    pickle.dump(all_accuracies_amp, f)

with open(fedavg_results_path, 'wb') as f:
    pickle.dump(all_accuracies_avg, f)

print(f"FedAMP accuracies saved to {fedamp_results_path}.")
print(f"FedAvg accuracies saved to {fedavg_results_path}.")


