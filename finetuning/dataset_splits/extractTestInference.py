import json
pth = r"C:\Users\marti\Music\knn\KNN_part_3\KNN_project\finetuning\dataset_splits\test_dataset.json"

import json

with open(pth, "r", encoding="utf-8") as pf:
        data = json.load(pf)

        # Extract only the 'content' fields
        content_list = [item['content'] for item in data]

        with open("test1infer.json", "w", encoding="utf-8") as wf:
            json.dump(content_list, wf, ensure_ascii=False, indent=4)