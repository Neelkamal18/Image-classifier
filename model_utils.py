import torch
from torch import nn, optim
from torchvision import models

def build_model(arch='vgg16', hidden_units=512, learning_rate=0.001):
    # Load a pre-trained model
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier based on the architecture
    if arch.startswith('vgg') or arch.startswith('densenet'):
        input_size = model.classifier[0].in_features if hasattr(model.classifier[0], 'in_features') else 25088
        classifier = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier
    elif arch.startswith('resnet'):
        input_size = model.fc.in_features
        classifier = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        model.fc = classifier

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    return model, criterion, optimizer

def train_model(model, dataloaders, criterion, optimizer, epochs=5, gpu=False):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    print_every = 16
    steps = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        if steps % print_every == 0:
            model.eval()
            val_loss = 0
            accuracy = 0

            with torch.no_grad():
                for inputs, labels in dataloaders['val']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model(inputs)
                    val_loss += criterion(logps, labels).item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                f"Training Loss: {running_loss/len(dataloaders['train']):.3f}.. "
                f"Validation Loss: {val_loss/len(dataloaders['val']):.3f}.. "
                f"Validation Accuracy: {accuracy/len(dataloaders['val']):.3f}")

def save_checkpoint(model, train_dataset, save_dir, arch, epochs, optimizer):
    checkpoint = {
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': train_dataset.class_to_idx,
        'epochs': epochs,
        'optimizer_state': optimizer.state_dict(),
        'arch': arch
    }
    save_path = f"{save_dir}/checkpoint.pth"
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    # Load pre-trained model
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {checkpoint['arch']}")

    if hasattr(model, 'classifier'):
        model.classifier = checkpoint['classifier']
    else:
        model.fc = checkpoint['classifier']

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def predict(image, model, topk=5, gpu=False):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Forward pass
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        logps = model(image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)

    # Map indices to classes
    top_p = top_p.cpu().numpy().flatten().tolist()
    top_class = top_class.cpu().numpy().flatten().tolist()
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[c] for c in top_class]

    return top_p, top_classes
