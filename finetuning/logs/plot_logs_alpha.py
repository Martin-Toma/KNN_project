import re
import matplotlib.pyplot as plt

# load logs
base_pth = r"C:\Users\marti\Music\knn\trainedModels\logs\\"
log_files = ['log10r32a16', 'log10r32a32', 'log6r32a64']
rename = { 
    'log10r32a16': "Rank 32 Alpha 16 Dropout 0.1", 
    'log10r32a32': "Rank 32 Alpha 32 Dropout 0.1", 
    'log6r32a64': "Rank 32 Alpha 64 Dropout 0.1",
}
# Initialize the figure outside the loop
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
    with open(pth, 'r') as f:
        log_text = f.read()

    log_entries = re.findall(r"\{[^}]+\}", log_text)
    epochs = []
    losses = []
    accuracies = []

    for entry in log_entries:
        try:
            epoch_match = re.search(r"'epoch': ([\d.]+)", entry)
            loss_match = re.search(r"'loss': ([\d.]+)", entry)
            acc_match = re.search(r"'mean_token_accuracy': ([\d.]+)", entry)

            if epoch_match and loss_match and acc_match:
                epochs.append(float(epoch_match.group(1)))
                losses.append(float(loss_match.group(1)))
                accuracies.append(float(acc_match.group(1)))
        except Exception as e:
            print(f"Skipping entry due to error: {e}")

    # Add plots to the subplots
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label=rename[file])

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label=rename[file])

# Show legends and finalize plot
plt.subplot(1, 2, 1)
plt.legend()

plt.subplot(1, 2, 2)
plt.legend()

plt.tight_layout()
plt.show()
