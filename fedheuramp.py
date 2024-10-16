def attention_derivative(wi, wj, sigma):
    cos_sim = np.dot(wi, wj) / (np.linalg.norm(wi) * np.linalg.norm(wj))
    return np.exp(sigma * cos_sim)

def aggregate_models_amp(clients, alpha, sigma, num_clients):
    new_models = []
    for i in range(num_clients):
        wi = np.concatenate([p.data.cpu().numpy().ravel() for p in clients[i].parameters()])
        weighted_sum = np.zeros_like(wi)
        total_weight = 0
        
        numerator = []
        for j in range(num_clients):
            if i != j:
                wj = np.concatenate([p.data.cpu().numpy().ravel() for p in clients[j].parameters()])
                weight = attention_derivative(wi, wj, sigma)
                numerator.append(weight)
        
        denominator = sum(numerator)
        normalized_weights = [w / denominator for w in numerator]

        print(f"Client {i}, Normalized Weights: {normalized_weights}")  # 추가된 출력 라인
        
        for j, norm_weight in enumerate(normalized_weights):
            wj = np.concatenate([p.data.cpu().numpy().ravel() for p in clients[j + 1 if j >= i else j].parameters()])
            weighted_sum += norm_weight * wj
            total_weight += norm_weight
        
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
