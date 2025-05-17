from datasets import load_dataset
dataset = load_dataset("json", data_files=r"C:\Users\marti\Music\knn\KNN_part_3\KNN_project\finetuning\dataset_splits\rev3_test_32.json")

print(dataset)
 
with open('test.out', 'w') as to:
    to.write(str(dataset['train'][4]))