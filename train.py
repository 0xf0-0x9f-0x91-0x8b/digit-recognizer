import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from tqdm.auto import trange
import numpy as np

import torch
import torch.nn.functional as F

from model import dump

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss/len(train_loader)

def evaluate(model, train_loader, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = correct / total
    
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = correct / total
    return train_acc, test_acc

def init_iplot(num_epochs):
    plt.close('all')
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 4))
    (train_line,) = ax.plot([], [], label="Train acc")
    (test_line,)  = ax.plot([], [], label="Test  acc")
    
    train_scatter = ax.scatter([], [], s=24, zorder=3)
    test_scatter  = ax.scatter([], [], s=24, zorder=3)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(1, num_epochs)   # fixed x range so the axes don't jump
    ax.set_ylim(0.0, 1.0)        # will adjust below as data comes in
    ax.legend(loc="lower right")
    display_handle = display(fig, display_id=True)
    return fig, ax, display_handle, train_line, test_line, train_scatter, test_scatter

def update_iplot(train_accuracies, test_accuracies, fig, ax, display_handle, train_line, test_line, train_scatter, test_scatter):
    x = np.arange(1, len(train_accuracies) + 1)

    train_line.set_data(x, train_accuracies)
    test_line.set_data(x, test_accuracies)
    train_scatter.set_offsets(np.column_stack([x, train_accuracies]))
    test_scatter.set_offsets(np.column_stack([x, test_accuracies]))

    # Safe manual y autoscale (avoid ax.relim/autoscale shape issues)
    yy = np.r_[train_accuracies, test_accuracies]
    ymin, ymax = float(np.min(yy)), float(np.max(yy))
    pad = max(0.02, 0.05 * max(ymax - ymin, 1e-6))
    ax.set_ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))

    fig.canvas.draw()
    display_handle.update(fig)
    
def run(num_epochs, model, optimizer, train_loader, test_loader):
    fig, ax, display_handle, train_line, test_line, train_scatter, test_scatter = init_iplot(num_epochs)
    train_accuracies, test_accuracies = [], []
    
    pbar = trange(1, num_epochs + 1, desc="Epochs", leave=True)
    best_test_acc = 0
    for epoch in pbar:
        loss = train(model, train_loader, optimizer)
        train_acc, test_acc = evaluate(model, train_loader, test_loader)
        if test_acc > best_test_acc:
            dump(model, 'results/model.safetensors')
            print(f'model dumped with test accuracy {test_acc}')
            best_test_acc = test_acc

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    
        pbar.set_postfix(
            loss=f"{loss:.4f}",
            train_acc=f"{train_acc:.3f}",
            test_acc=f"{test_acc:.3f}"
        )
        update_iplot(train_accuracies, test_accuracies, fig, ax, display_handle, train_line, test_line, train_scatter, test_scatter)