import argparse
from model_utils import build_model, train_model, save_checkpoint
from data_utils import load_data
import os

def main():
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset.")
    parser.add_argument("data_dir", type=str, help="Directory of the dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of images in a batch")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="vgg16", help="Model architecture")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()
    
    # Learning Rate Validation
    if args.learning_rate <= 0 or args.learning_rate > 1:
        raise ValueError("Learning rate must be a positive number between 0 and 1.")

    # Save Directory Check
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load data
    dataloaders, image_datasets = load_data(args.data_dir)
    
    # Build model
    model, criterion, optimizer = build_model(args.arch, args.hidden_units, args.learning_rate)
    
    # Train model
    train_model(model, dataloaders, criterion, optimizer, args.epochs, args.gpu)
    
    # Save checkpoint
    save_checkpoint(model, image_datasets['train'], args.save_dir, args.arch, args.epochs, optimizer)
    
if __name__ == "__main__":
    main()
    
# Example usage
# python3 train.py data_directory --batch_size 64 --save_dir save_directory --arch "vgg13" --learning_rate 0.01 --hidden_units 512 --epochs 20 --gpu
