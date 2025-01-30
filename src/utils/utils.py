import copy
import os
import pickle
import argparse
import numpy as np
from typing import List
from PIL import Image
import torch
from torchvision import models
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, Features, ClassLabel
from datasets import Image as DatasetImage
from torchvision import datasets, transforms
import fcntl

from preprocess.preConfig import PreConfig

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner, DirichletPartitioner, IidPartitioner

def parse_args():
    argparser = argparse.ArgumentParser()

    # FL Parameters
    argparser.add_argument("--N", type=int, default=100, help="Number of users")
    argparser.add_argument("--p", type=int, default=10, help="Number of clients")
    argparser.add_argument("--E", type=int, default=5, help="Number of local update steps")
    argparser.add_argument("--num-rounds", type=int, default=10000, help="Number of FL rounds")
    argparser.add_argument("--alpha", type=float, default=0.1, help="Degree of non-iid, iid if -1")

    # Training Parameters
    argparser.add_argument("--B", type=int, default=32, help="Batch Size")
    argparser.add_argument("--target-accuracy", type=float, default=1, help="Target accuracy to stop training")
    argparser.add_argument("--spi", type=float, default=0.1, help="Seconds per iteration")

    argparser.add_argument("--method", type=str, default='Random', help="What Method to use: [Balance, IID, Non-IID, Random, FedProx]")

    # Critic Parameters
    argparser.add_argument("--num-samples", type=int, default=100, help="Number of samples to estimate parameters")
    argparser.add_argument("--epsilon", type=float, default=0.01, help="Threshold of grad_norm_squred that defines convergence")

    # Actor Parameters
    argparser.add_argument("--delta", type=float, default=0.1, help="Size of datapoint")

    argparser.add_argument("--save-dir", type=str, default="data", help="Directory to save data")
    argparser.add_argument("--Pass2CenterRation", type=float, default=0.2, help="Directory to save data")


    args = argparser.parse_args()
    print(args)
    return args


def partition_cifar10(distribution):
    dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    labels = np.array(dataset.targets)
    class_indices = {class_id: np.where(labels == class_id)[0] for class_id in range(10)}
    for class_id in class_indices:
        np.random.shuffle(class_indices[class_id])
    partitions = {}

    # Define feature schema for CIFAR-10
    feature_schema = Features({
        'img': DatasetImage(),
        'label': ClassLabel(
            names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        )
    })

    for i, counts in enumerate(distribution):
        partition_indices = []
        for class_id, num_samples in enumerate(counts):
            partition_indices.extend(class_indices[class_id][:num_samples])
            class_indices[class_id] = class_indices[class_id][num_samples:]
        np.random.shuffle(partition_indices)
        partition_data = [
            {"img": Image.fromarray(dataset.data[idx]), "label": int(labels[idx])} for idx in partition_indices
        ]
        partitions[i] = Dataset.from_list(partition_data)

        # Attach the schema to the partition
        partitions[i] = partitions[i].cast(feature_schema)
    return partitions



def load_datasets(save_dir: str, partition_id: int, num_partitions: int, batch_size: int, alpha: float = 0.1, distribution=None):
    def apply_pytorch_transforms(batch):
        pytorch_transforms = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch
    os.makedirs(save_dir, exist_ok=True)
    # Partition dataset for users using the flwr partitioner
    dataset_dir = os.path.join(save_dir, "fds.pkl")
    # Only create it once since multiple calls would give a different dataset due to randomness
    if not os.path.exists(dataset_dir) or alpha == -1:
        if alpha == -1:
            # IID Partitioning: Each client gets a random subset of the entire dataset
            partitioner = {"train": IidPartitioner(num_partitions)}  # Default behavior for IID
        else:
            # Non-IID Partitioning: Skew data based on labels or some other strategy
            dirichlet_partitioner = DirichletPartitioner(
                num_partitions, alpha=alpha, partition_by="label"
            )
            partitioner = {"train": dirichlet_partitioner}  # Example of label-skewed partitioning
        fds = FederatedDataset(dataset="cifar10", partitioners=partitioner)
        if alpha != -1:
            with open(dataset_dir, "wb") as f:
                pickle.dump(fds, f)
    else:
        with open(dataset_dir, "rb") as f:
            fds = pickle.load(f)
    partition = fds.load_partition(partition_id)

    labels = partition["label"]

    # Count unique labels and their occurrences
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Print results
    for label, count in zip(unique_labels, counts):
        print(f"Label: {label}, Count: {count}")

    testset = fds.load_split("test").with_transform(apply_pytorch_transforms)
    if distribution is not None:
        # Partition dataset for clients using the defined distribution
        dataset_dir = os.path.join(save_dir, "fds_distribution.pkl")
        if not os.path.exists(dataset_dir):
            fds = partition_cifar10(distribution)
            with open(dataset_dir, "wb") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock
                pickle.dump(fds, f)
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock
        else:
            with open(dataset_dir, "rb") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                fds = pickle.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)
        partition = fds[partition_id]

    num_classes = len(partition.features['label'].names)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    partition_train_test = partition_train_test.with_transform(apply_pytorch_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, valloader, testloader, num_classes

def Net():
    model = models.resnet18(weights=None)
    model.train()
    return model

def run_pre_experiments(p, B, alpha, num_samples, initial_model, criterion, save_dir, device):
    configs = []
    # num_classes = 10
    # for partition_id in range(p):
    #     dataset = load_datasets(save_dir, partition_id, p, B, alpha)[0]
    #     preConfig = PreConfig(Net, criterion, dataset, num_samples, num_classes, device)
    #     L = preConfig.get_L()
    #     C1, sigma_squared = preConfig.get_C1_sigma(copy.deepcopy(initial_model), B)
    #     loss = preConfig.get_loss(initial_model)
    #     average_gradients, class_counts = preConfig.get_gradient(copy.deepcopy(initial_model))
    #     configs.append((float(L), float(C1), float(sigma_squared), float(loss), average_gradients, class_counts))

    dataset, _, _, num_classes = load_datasets(save_dir, 0, p, B, alpha=-1)
    preConfig = PreConfig(Net, criterion, dataset, num_samples, num_classes, device)
    L = preConfig.get_L()
    C1, sigma_squared = preConfig.get_C1_sigma(copy.deepcopy(initial_model), B)
    loss = preConfig.get_loss(initial_model)

    for partition_id in range(p):
        dataset = load_datasets(save_dir, partition_id, p, B, alpha)[0]
        preConfig = PreConfig(Net, criterion, dataset, num_samples, num_classes, device)
        average_gradients, class_counts = preConfig.get_gradient(copy.deepcopy(initial_model))
        configs.append((float(L), float(C1), float(sigma_squared), float(loss), average_gradients, class_counts))

    # Zeta
    mix_matrix = np.ones((p, p)) # Assume fully connected
    eigenvalues = np.linalg.eigvals(mix_matrix)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    zeta = sorted_eigenvalues[1] if len(sorted_eigenvalues) > 1 else 0 # Second largest eigenvalue, which is zero when graoh is fully connected
    return configs, float(zeta), num_classes

def set_parameters(net, parameters: List[np.ndarray]):
    state_dict = net.state_dict()
    for (name, param), np_param in zip(state_dict.items(), parameters):
        state_dict[name] = torch.tensor(np_param)
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def train(net, train_iter, train_loader, local_iterations: int, lr: float, device, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    total_gradient_norm_squared = []
    losses = []
    for _ in range(local_iterations):
        try:
            batch = next(train_iter)  # Get the next batch
        except:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images, labels = batch["img"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        grad_norm_squared = 0.0
        for param in net.parameters():
            if param.grad is not None:
                grad_norm_squared += param.grad.norm(2).item() ** 2
        total_gradient_norm_squared.append(grad_norm_squared)
        losses.append(loss.detach().item())
        optimizer.step()
    return total_gradient_norm_squared, sum(losses)/len(losses), train_iter

def test(net, testloader, device):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy