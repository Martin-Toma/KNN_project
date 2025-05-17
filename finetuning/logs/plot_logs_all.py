import re
import matplotlib.pyplot as plt

# load logs
base_pth = r"C:\Users\marti\Music\knn\trainedModels\logs\\"
log_files = ['log6dropout2', 'log6dropout05', 'log6qlora', 'log10r8a16', 'log10r16a32', 'log10r32a16', 'log10r32a32', 'log6r32a64']

for file in log_files:
    pth = base_pth + file + ".txt"

    f = open(pth, 'r') #f = open("./logDropout2", "r")
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