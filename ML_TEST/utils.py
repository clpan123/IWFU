import torch
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def return_acc(model,dataloader):
    model.eval()
    model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        acc = round(correct/total,3)
    # print("本次测试精度为:",acc)
    return acc