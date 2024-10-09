import numpy as np
import torch

def attention_derivative(wi, wj, sigma):
    diff = np.linalg.norm(wi - wj)**2
    return np.exp(-diff / sigma)

def aggregate_models_amp(clients, alpha, sigma, num_clients):
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

def aggregate_models_avg(clients, num_clients):
    avg_model = clients[0].state_dict()
    for key in avg_model.keys():
        avg_model[key] = torch.stack([clients[i].state_dict()[key] for i in range(num_clients)], dim=0).mean(dim=0)
    return avg_model
