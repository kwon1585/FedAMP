import torch
import numpy as np

def local_update(model, cloud_model, optimizer, criterion, dataloader, alpha=None, lambda_param=None, device='cpu'):
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

def evaluate(model, dataloader, criterion, device='cpu'):
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
