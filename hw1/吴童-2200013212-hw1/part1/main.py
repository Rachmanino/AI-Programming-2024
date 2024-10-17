import torch
import torch.utils.data
import torchvision
from data import get_dataloader
from model import LeNet
import json
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

batchsize = 32
epochs = 10
lr = 1e-2
momentum = 0.95

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = './model.pth'

classes = 10

def train(model, train_loader,  criterion, optimizer):
    model.train()
    for epoch in tqdm(range(epochs)):
        losses = []
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
        
        print(f'Loss: {sum(losses) / len(losses)}')
        writer.add_scalar('Loss/train', sum(losses) / len(losses), epoch)

def test(model, test_loader):
    model.eval()
    correct = torch.zeros(classes, device=device)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            correct += (F.one_hot(logits.argmax(1), classes) & F.one_hot(labels, classes)).sum(0)

    with open('results.json', 'w') as f:
        json.dump({cls: correct[i].item() / (len(test_loader.dataset)/10) for i, cls in enumerate(train_loader.dataset.classes)}, f)
    print(f'Accuracy: {correct.sum().item() / len(test_loader.dataset)}')

    

if __name__ == '__main__':
    train_loader, test_loader = get_dataloader(batchsize)
    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, criterion, optimizer)
    torch.save(model, model_path)
    model = torch.load(model_path)
    test(model, test_loader)
    writer.flush()
    writer.close()

    
