import torch
import torch.nn as nn
from data_pipeline import load_data
from model import MNIST_CNN

train_loader, val_loader, test_loader = load_data()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

criterion = nn.CrossEntropyLoss()
model = MNIST_CNN().to(device)
criterion = criterion.to(device)

import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 5

# train
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    train_acc = correct / total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = criterion(out, y)

            val_loss += loss.item()
            _, pred = out.max(1)
            val_correct += pred.eq(y).sum().item()
            val_total += y.size(0)
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | "
          f"Val Acc: {val_acc:.4f}")

# evaluation
model.eval()
eval_correct = 0
eval_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        _, pred = out.max(1)

        eval_correct += pred.eq(y).sum().item()
        eval_total += y.size(0)

        all_preds.append(pred.cpu())
        all_labels.append(y.cpu)
eval_acc = eval_correct / eval_total
print(f"Test Accuracy: {eval_acc:.4f}")


import random
import matplotlib.pyplot as plt

model.eval()
images = []
labels = []
preds = []
grid = 8

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        _, p = out.max(1)

        images.append(x.cpu())
        labels.append(y)
        preds.append(p.cpu())

images = torch.cat(images)
labels = torch.cat(labels)
preds  = torch.cat(preds)

idx = random.sample(range(len(images)), grid*grid)

plt.figure(figsize=(12,12))
for i, j in enumerate(idx):
    plt.subplot(grid, grid, i+1)
    plt.imshow(images[j][0], cmap="gray")
    plt.title(f"P:{preds[j].item()} T:{labels[j].item()}", fontsize=8)
    plt.axis("off")
plt.show()