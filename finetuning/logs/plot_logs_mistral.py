import re
import ast
import matplotlib.pyplot as plt

# Load logs
base_pth = r"C:\Users\marti\Music\knn\trainedModels\logs\\"
log_files = [
    'mistral03', 'mistral01'
]
rename = {
    'mistral01': "Rank 16 Alpha 32 Dropout 0.1 Max grad norm 0.1", 
    'mistral03': "Rank 16 Alpha 32 Dropout 0.1 Max grad norm 0.3",
}

# Initialize the figure
plt.figure(figsize=(14, 6))

# Prepare subplots for loss and accuracy
plt.subplot(1, 2, 1)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title("Mean Token Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

# Process each file
for file in log_files:
    pth = base_pth + file + ".txt"
    try:
        with open(pth, 'r') as f:
            log_text = f.read()
    except FileNotFoundError:
        print(f"File not found: {pth}")
        continue

    # Clean whitespace and flatten multiline entries
    log_text = re.sub(r'\s+', ' ', log_text)

    # Extract dict-like log entries
    log_entries = re.findall(r"\{[^}]+\}", log_text)
    epochs = []
    losses = []
    accuracies = []

    for entry in log_entries:
        try:
            parsed = ast.literal_eval(entry)
            epoch = parsed.get('epoch')
            loss = parsed.get('loss')
            acc = parsed.get('mean_token_accuracy')

            if epoch is not None and loss is not None and acc is not None:
                epochs.append(float(epoch))
                losses.append(float(loss))
                accuracies.append(float(acc))
        except Exception as e:
            print(f"Skipping malformed entry in {file}: {e}")

    # Sort by epoch to ensure smooth line plots
    if epochs:
        sorted_data = sorted(zip(epochs, losses, accuracies))
        epochs, losses, accuracies = zip(*sorted_data)

        # Add plots to the subplots
        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, label=rename[file])

        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracies, label=rename[file])
    else:
        print(f"No valid data parsed from {file}")

# Finalize plots
plt.subplot(1, 2, 1)
plt.legend()

plt.subplot(1, 2, 2)
plt.legend()

plt.tight_layout()
plt.show()