import re
import matplotlib.pyplot as plt

# Paste your log as a multi-line string here
f = open("./logDropout2", "r")
log_text = f.read()
f.close()

# Extract individual training records using regex
log_entries = re.findall(r"\{[^}]+\}", log_text)

# Initialize lists to store extracted values
epochs = []
losses = []
accuracies = []

# Iterate through each JSON-like entry and extract metrics
for entry in log_entries:
    try:
        # Extract values using regex
        epoch_match = re.search(r"'epoch': ([\d.]+)", entry)
        loss_match = re.search(r"'loss': ([\d.]+)", entry)
        acc_match = re.search(r"'mean_token_accuracy': ([\d.]+)", entry)

        if epoch_match and loss_match and acc_match:
            epochs.append(float(epoch_match.group(1)))
            losses.append(float(loss_match.group(1)))
            accuracies.append(float(acc_match.group(1)))
    except Exception as e:
        print(f"Skipping entry due to error: {e}")

# Plotting
plt.figure(figsize=(12, 6))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, losses, marker='o', color='red')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracies, marker='o', color='green')
plt.title("Mean Token Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()
plt.show()