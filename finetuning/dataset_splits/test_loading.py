from datasets import load_dataset

dataset = load_dataset("Nirmata/Movie_evaluation")
print(dataset["validation"][0])

"""
data_files={
        "train": "https://huggingface.co/datasets/Nirmata/Movie_evaluation/resolve/main/train.json",
        "test": "https://huggingface.co/datasets/Nirmata/Movie_evaluation/resolve/main/test.json",
        "validation": "https://huggingface.co/datasets/Nirmata/Movie_evaluation/resolve/main/validation.json",
    }
"""