from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(data_dir, batch_size=64):
    """Load and preprocess data for training, validation, and testing."""
    # Define transforms for training, validation, and testing
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(root=f"{data_dir}/train", transform=data_transforms['train']),
        'val': datasets.ImageFolder(root=f"{data_dir}/valid", transform=data_transforms['val']),
        'test': datasets.ImageFolder(root=f"{data_dir}/test", transform=data_transforms['test'])
    }

    # Create dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False)
    }

    print(f"Data successfully loaded from {data_dir}.")
    print(f"Training set: {len(image_datasets['train'])} images, Validation set: {len(image_datasets['val'])} images, Test set: {len(image_datasets['test'])} images.")

    return dataloaders, image_datasets