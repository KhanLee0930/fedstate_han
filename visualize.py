import numpy as np
import os
import matplotlib.pyplot as plt

ROOT = "/Users/apple/Desktop/fedstate/output/convergence_results/30-10-10-0.1-32"
data_sharing_methods = ["data_sharing-0-0", "data_sharing-0-10", "data_sharing-0-20"]
fedprox_methods = ["fedprox-0-0", "fedprox-0-10", "fedprox-0-20"]

fig, axes = plt.subplots(len(data_sharing_methods), 3, figsize=(30, 20))  # 3行2列

for i, method in enumerate(data_sharing_methods):
    path = os.path.join(ROOT, method)
    test_acc = np.load(os.path.join(path, "test_acc.npy"))
    train_loss = np.load(os.path.join(path, "train_loss.npy"))
    gns = np.load(os.path.join(path, "actual_GNS.npy"))

    epochs = np.arange(len(test_acc))
    axes[i, 0].plot(epochs, test_acc, label="Test Accuracy", color="blue", linewidth=1, alpha=0.8)
    axes[i, 0].set_xlabel("Epochs", fontsize=20)
    axes[i, 0].set_ylabel("Test Accuracy", fontsize=20)
    axes[i, 0].set_title(f"{method} - Test Accuracy", fontsize=20)
    axes[i, 0].legend(fontsize=20, loc="upper right")
    axes[i, 0].grid(True, linestyle="--", alpha=0.6)

    train_epochs = np.arange(len(train_loss))
    axes[i, 1].plot(train_epochs, train_loss, label="Train Loss",color="blue", linewidth=1, alpha=0.8,)
    axes[i, 1].set_xlabel("Epochs", fontsize=20)
    axes[i, 1].set_ylabel("Train Loss", fontsize=20)
    axes[i, 1].legend(fontsize=20, loc="upper right")
    axes[i, 1].grid(True, linestyle="--", alpha=0.6)

    gns_epochs = np.arange(len(gns))
    axes[i, 2].plot(gns_epochs, gns, label="GNS",color="blue", linewidth=1, alpha=0.8,)
    axes[i, 2].set_xlabel("Epochs", fontsize=20)
    axes[i, 2].set_ylabel("GNS", fontsize=20)
    axes[i, 2].legend(fontsize=20, loc="upper right")
    axes[i, 2].grid(True, linestyle="--", alpha=0.6)

for i, method in enumerate(fedprox_methods):
    path = os.path.join(ROOT, method)
    test_acc = np.load(os.path.join(path, "test_acc.npy"))
    train_loss = np.load(os.path.join(path, "train_loss.npy"))
    gns = np.load(os.path.join(path, "actual_GNS.npy"))

    epochs = np.arange(len(test_acc))
    axes[i, 0].plot(epochs, test_acc, label="FedProx Test Acc", color="red", linewidth=1, alpha=0.8)
    axes[i, 0].legend(fontsize=20, loc="upper right")

    train_epochs = np.arange(len(train_loss))
    axes[i, 1].plot(train_epochs, train_loss, label="FedProx Train Loss",
                    color="red", linewidth=1, alpha=0.8)
    axes[i, 1].legend(fontsize=20, loc="upper right")

    gns_epochs = np.arange(len(gns))
    axes[i, 2].plot(gns_epochs, gns, label="GNS",color="red", linewidth=1, alpha=0.8,)
    axes[i, 2].legend(fontsize=20, loc="upper right")

plt.tight_layout()  # 自动调整布局，防止重叠
plt.savefig("training_results.png", dpi=300, bbox_inches="tight")  # 保存高分辨率图片
plt.show()
