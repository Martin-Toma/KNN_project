from pathlib import Path
import pandas as pd
import numpy as np

folder = 'C:\\Users\\marti\\Music\\knn\\KNN_project\\model_inference\\perplexities'

ppls = []

for i in range(0, 1000):
    try:
        inFile = Path(folder + f"\\{i}.txt").resolve()
        with open(inFile, 'r', encoding='utf-8') as inf:
            ppl_value = float(inf.read().strip())
            if ppl_value > 0 and np.isfinite(ppl_value):
                ppls.append(ppl_value)
            print(ppl_value)
        
    except Exception as e:
        print(e)

print(ppls)

df = pd.DataFrame(ppls)

# Calling the methods to compute the statistics
min_ppl = df.min()
max_ppl = df.max()
avg_ppl = df.mean()
std_ppl = df.std()

print(f"Perplexity statistics are:")
print(f"Minimum: {min_ppl.iloc[0]}")
print(f"Maximum: {max_ppl.iloc[0]}")
print(f"Average: {avg_ppl.iloc[0]}")
print(f"Standard deviation: {std_ppl.iloc[0]}")