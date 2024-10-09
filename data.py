import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

def create_client_datasets(train_dataset, num_clients, test_size_per_client):
    clients_data = [[] for _ in range(num_clients)]
    digit_clients = list(range(10))
    upper_clients = list(range(10, 36))
    lower_clients = list(range(36, 62))
    
    digit_indices = [idx for idx, (data, target) in enumerate(train_dataset) if target < 10]
    upper_indices = [idx for idx, (data, target) in enumerate(train_dataset) if 10 <= target < 36]
    lower_indices = [idx for idx, (data, target) in enumerate(train_dataset) if 36 <= target < 62]

    for idx, (data, target) in enumerate(train_dataset):
        if target < 10:
            clients_data[np.random.choice(digit_clients)].append(idx)
        elif 10 <= target < 36:
            clients_data[np.random.choice(upper_clients)].append(idx)
        elif 36 <= target < 62:
            clients_data[np.random.choice(lower_clients)].append(idx)

    return clients_data, digit_indices, upper_indices, lower_indices

def assign_data_with_dominating_and_other_classes(train_dataset, clients_data, digit_indices, upper_indices, lower_indices, dominant_ratio=0.8, test_size_per_client=100):
    client_train_datasets = []
    client_test_datasets = []

    for indices in clients_data:
        dominant_size = int(len(indices) * dominant_ratio)
        other_size = len(indices) - dominant_size
        dominant_indices = indices[:dominant_size]
        
        dominant_class = train_dataset[dominant_indices[0]][1]
        if dominant_class < 10:
            available_other_indices = upper_indices + lower_indices
        elif 10 <= dominant_class < 36:
            available_other_indices = digit_indices + lower_indices
        else:
            available_other_indices = digit_indices + upper_indices

        other_indices = np.random.choice(available_other_indices, size=other_size, replace=False).tolist()
        
        all_indices = dominant_indices + other_indices
        np.random.shuffle(all_indices)

        train_indices, test_indices = random_split(all_indices, [len(all_indices) - test_size_per_client, test_size_per_client])
        client_train_datasets.append(Subset(train_dataset, train_indices))
        client_test_datasets.append(Subset(train_dataset, test_indices))

    return client_train_datasets, client_test_datasets

def create_dataloaders(train_dataset, num_clients, test_size_per_client, batch_size):
    clients_data, digit_indices, upper_indices, lower_indices = create_client_datasets(train_dataset, num_clients, test_size_per_client)
    client_train_datasets, client_test_datasets = assign_data_with_dominating_and_other_classes(train_dataset, clients_data, digit_indices, upper_indices, lower_indices)

    client_dataloaders = [DataLoader(client_train_datasets[i], batch_size=batch_size, shuffle=True) for i in range(num_clients)]
    client_test_dataloaders = [DataLoader(client_test_datasets[i], batch_size=batch_size, shuffle=False) for i in range(num_clients)]

    return client_dataloaders, client_test_dataloaders

def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)
    test_dataset = datasets.EMNIST(root='./data', split='byclass', train=False, download=True, transform=transform)

    return train_dataset, test_dataset
