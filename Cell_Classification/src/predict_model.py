import torch

def predict_local(model, data_loader, calculate_custom_score, device) -> int:
    model.eval()
    total_0 = 0
    total_1 = 0
    correct_0 = 0
    correct_1 = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # Calculate correct predictions and totals for each class
            total_0 += (labels == 0).sum().item()
            total_1 += (labels == 1).sum().item()

            correct_0 += ((predicted == 0) & (labels == 0)).sum().item()
            correct_1 += ((predicted == 1) & (labels == 1)).sum().item()
            
            print(f'labels:{labels}')
            print(f'predicted:{predicted}')

    # Calculate the custom score
    custom_score = calculate_custom_score(correct_0, correct_1, total_0, total_1)
    
    return custom_score