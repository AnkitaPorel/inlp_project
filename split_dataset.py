import json
import os
import torch
import random


def split_dataset(
    input_file, train_output_file, test_output_file, test_size=0.2, random_state=42
):
    """
    Split a dataset into training and testing sets using PyTorch.

    Args:
        input_file (str): Path to the input JSON file
        train_output_file (str): Path to save the training data
        test_output_file (str): Path to save the testing data
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
    """
    # Set seed for reproducibility
    torch.manual_seed(random_state)
    random.seed(random_state)

    # Load the data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Calculate split sizes
    dataset_size = len(data)
    test_count = int(dataset_size * test_size)
    train_count = dataset_size - test_count

    # Create a random split with PyTorch
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_count]
    test_indices = indices[train_count:]

    # Create train and test datasets
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]

    # Save the training data
    with open(train_output_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    # Save the testing data
    with open(test_output_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")


def main():
    # Define file paths
    input_file = os.path.join("data", "poems.json")
    train_output_file = os.path.join("data", "train.json")
    test_output_file = os.path.join("data", "test.json")

    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(train_output_file), exist_ok=True)

    # Split the dataset
    split_dataset(input_file, train_output_file, test_output_file)


if __name__ == "__main__":
    main()
