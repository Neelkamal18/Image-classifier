import argparse
from model_utils import load_checkpoint, predict
from image_utils import process_image
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Predict the class of an input image.")
    parser.add_argument("image_path", type=str, help="Path to the image")
    parser.add_argument("checkpoint", type=str, help="Model checkpoint to use for prediction")
    parser.add_argument("--top_k", type=int, default=1, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to category-to-name JSON mapping")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    args = parser.parse_args()

    # File existence checks
    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    if args.category_names and not os.path.isfile(args.category_names):
        raise FileNotFoundError(f"Category names file not found: {args.category_names}")

    # Load the checkpoint
    model = load_checkpoint(args.checkpoint)

    # Process the image
    image = process_image(args.image_path)

    # Make prediction
    probs, classes = predict(image, model, args.top_k, args.gpu)

    # Map categories to names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name.get(cls, cls) for cls in classes]

    # Print results
    print("Top Predictions:")
    for i, (cls, prob) in enumerate(zip(classes, probs), start=1):
        print(f"{i}. {cls}: {prob:.2%}")

if __name__ == "__main__":
    main()

# Example usage
# python3 predict.py /path/to/image checkpoint --top_k 3 --category_names cat_to_name.json --gpu
