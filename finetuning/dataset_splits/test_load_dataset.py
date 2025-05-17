from datasets import load_dataset
#from pathlib import Path

path_validation = r'C:\Users\marti\Music\knn\KNN_project_part3\KNN_project\finetuning\dataset_splits\instruction_eval_dataset.json'
#rp = Path(path).resolve()

data_files_validation = {
    "validation": path_validation,
}

validation_dataset = load_dataset("json", data_files=data_files_validation)

print(validation_dataset['validation'][0])