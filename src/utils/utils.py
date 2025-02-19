import logging
# Suppress warnings from the Hugging Face datasets library
logging.getLogger("datasets").setLevel(logging.ERROR)
# Suppress warnings from flwr_datasets
logging.getLogger("flwr_datasets").setLevel(logging.ERROR)
from scipy.special import psi
from scipy.optimize import minimize

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
import random
from preprocess.preConfig import PreConfig

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner, DirichletPartitioner, IidPartitioner

def parse_args():
    argparser = argparse.ArgumentParser()

    # FL Parameters
    argparser.add_argument("--N", type=int, default=50, help="Number of users")
    argparser.add_argument("--p", type=int, default=10, help="Number of clients")
    argparser.add_argument("--E", type=int, default=10, help="Number of local update steps")
    argparser.add_argument("--num-rounds", type=int, default=2000, help="Number of FL rounds")
    argparser.add_argument("--alpha", type=float, default=0.1, help="Degree of non-iid, iid if -1")
    argparser.add_argument("--delta", type=float, default=0.1, help="Size of datapoint")

    # Training Parameters
    argparser.add_argument("--B", type=int, default=32, help="Batch Size")
    argparser.add_argument("--seed", type=int, default=0, help="Random Seed")
    argparser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight Decay")

    # Method Parameters
    argparser.add_argument("--method", type=str, default='Balance', help="What Method to use: [Balance, IID, Non-IID, Random, FedProx]")
    argparser.add_argument("--balance-params", type=int, default=0, help="Whether to additionally balance the batch size and local epochs")
    argparser.add_argument("--num-actor-rounds", type=int, default=20001, help="Number of actor optimization rounds")

    # Critic Parameters
    argparser.add_argument("--num-samples", type=int, default=100, help="Number of samples to estimate parameters")
    argparser.add_argument("--epsilon", type=float, default=0.01, help="Threshold of grad_norm_squred that defines convergence")

    argparser.add_argument("--save-dir", type=str, default="data", help="Directory to save data")
    args = argparser.parse_args()
    print(args)
    return args


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def apply_pytorch_transforms(batch):
    pytorch_transforms = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_datasets(save_dir: str, partition_id: int, num_partitions: int, batch_size: int, alpha: float = 0.5):
    os.makedirs(save_dir, exist_ok=True)
    # Partition dataset for users using the flwr partitioner
    dataset_dir = os.path.join(save_dir, f"fds-{alpha}.pkl")
    # Only create it once since multiple calls would give a different dataset due to randomness
    if not os.path.exists(dataset_dir):
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
        with open(dataset_dir, "wb") as f:
            pickle.dump(fds, f)
    else:
        with open(dataset_dir, "rb") as f:
            fds = pickle.load(f)

    trainpartition = fds.load_partition(partition_id).with_transform(apply_pytorch_transforms)
    num_classes = len(trainpartition.features['label'].names)
    testset = fds.load_split("test").with_transform(apply_pytorch_transforms)

    trainloader = DataLoader(
        trainpartition, batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, testloader, num_classes


def load_datasets_distribution(save_dir: str, partition_id: int, batch_size: int, distribution=None):
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

    partitiontrain = fds[partition_id].with_transform(apply_pytorch_transforms)
    num_classes = len(partitiontrain.features['label'].names)
    trainloader = DataLoader(
        partitiontrain, batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(FederatedDataset(dataset="cifar10", partitioners={"train": IidPartitioner(10)}).load_split("test").with_transform(apply_pytorch_transforms), batch_size=batch_size)
    return trainloader, testloader, num_classes

def Net():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust for CIFAR10 classes
    model.train()
    return model

def run_pre_experiments(p, B, num_samples, initial_model, criterion, save_dir, device):
    # Obtain the model size in terms of GigaBytes
    model_size = 0
    for name, param in initial_model.named_parameters():
        model_size += param.numel() * param.element_size()
    model_size = model_size / (1024**3)

    # Load IID Dataset
    dataset, _, num_classes = load_datasets(save_dir, 0, p, B, alpha=-1)

    # Run Pre-Experiments, giving us L, C1, sigma_squared, loss, zeta, spi_predictor, and average gradients
    preConfig = PreConfig(Net, criterion, dataset, num_samples, num_classes, device)
    L = preConfig.get_L()
    C1, sigma_squared = preConfig.get_C1_sigma(copy.deepcopy(initial_model), B)
    loss = preConfig.get_loss(initial_model)
    spi_predictor = preConfig.get_spi(initial_model)
    average_gradients, _ = preConfig.get_gradient(copy.deepcopy(initial_model))
    # Zeta
    mix_matrix = np.ones((p, p)) # Assume fully connected
    eigenvalues = np.linalg.eigvals(mix_matrix)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    zeta = sorted_eigenvalues[1] if len(sorted_eigenvalues) > 1 else 0 # Second largest eigenvalue, which is zero when graoh is fully connected

    configs = (float(L), float(C1), float(sigma_squared), float(loss), float(zeta), average_gradients)
    return configs, num_classes, model_size, spi_predictor


def obtain_initial_distribution(p, alpha, save_dir):
    distribution = []
    for partition_id in range(p):
        dataset, _, num_classes = load_datasets(save_dir, partition_id, p, 32, alpha)
        class_counts = {cls: 0 for cls in range(num_classes)}
        for batch in dataset:
            for cls in range(num_classes):
                mask = (batch["label"] == cls)
                class_counts[cls] += mask.sum().item()
        class_counts = torch.tensor([class_counts[cls] for cls in range(num_classes)])
        distribution.append(class_counts)
    distribution = torch.stack(distribution)
    return distribution

def set_parameters(net, parameters: List[np.ndarray]):
    state_dict = net.state_dict()
    for (name, param), np_param in zip(state_dict.items(), parameters):
        state_dict[name] = torch.tensor(np_param)
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def train(net, optimizer, train_iter, train_loader, local_iterations: int, device):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    total_gradient_norm_squared = []
    losses = []
    for _ in range(local_iterations):
        try:
            batch = next(train_iter)  # Get the next batch
        except:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Update model
        images, labels = batch["img"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Log metrics
        grad_norm_squared = 0.0
        for param in net.parameters():
            if param.grad is not None:
                grad_norm_squared += param.grad.norm(2).item() ** 2
        total_gradient_norm_squared.append(grad_norm_squared)
        losses.append(loss.detach().item())

    return total_gradient_norm_squared, losses, train_iter

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