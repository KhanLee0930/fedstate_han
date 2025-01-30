from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset 
from flwr_datasets.partitioner import PathologicalPartitioner, DirichletPartitioner
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

os.environ["HF_HOME"] = "/scratch/e1310988"
os.environ['HF_DATASETS_CACHE'] = "/scratch/e1310988"

BATCH_SIZE = 32

def load_datasets(partition_id: int, num_partitions: int, iid: bool = False):
    if iid:
        # IID Partitioning: Each client gets a random subset of the entire dataset
        partitioner = {"train": num_partitions}  # Default behavior for IID
    else:
        # Non-IID Partitioning: Skew data based on labels or some other strategy
        pathological_partitioner = PathologicalPartitioner(
                num_partitions, partition_by="label", num_classes_per_partition=2
            )
        dirichlet_partitioner = DirichletPartitioner(
            num_partitions, alpha=0.1, partition_by="label"
        )
        partitioner = {"train": dirichlet_partitioner}   # Example of label-skewed partitioning

    fds = FederatedDataset(dataset="cifar10", partitioners=partitioner)
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


class FlowerClient(NumPyClient):
    def __init__(self, pid, net, trainloader, valloader):
        self.pid = pid  # partition ID of a client
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.pid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]

        # Use values provided by the config
        print(f"[Client {self.pid}, round {server_round}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=local_epochs)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.pid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn_wrapper(iid: bool):
    def client_fn(context: Context) -> Client:
        net = Net().to(DEVICE)
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        # Pass the iid argument to load_datasets
        trainloader, valloader, _ = load_datasets(partition_id, num_partitions, iid=iid)
        return FlowerClient(partition_id, net, trainloader, valloader).to_client()
    
    return client_fn


# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}

params = get_parameters(Net())

def evaluate(
    server_round: int,
    parameters: NDArrays,
    config: Dict[str, Scalar],
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    net = Net().to(DEVICE)
    _, _, testloader = load_datasets(0, NUM_PARTITIONS)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader)
    return loss, {"accuracy": accuracy}

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        # "local_epochs": 1 if server_round < 2 else 2,
        "local_epochs": 5,
    }
    return config

class CommunicationTrackingStrategy(FedAvg):
    def __init__(self, target_accuracy, **kwargs):
        super().__init__(**kwargs)
        self.total_communication_cost = 0
        self.target_accuracy = target_accuracy
        self.current_accuracy = 0.0

    def evaluate(self, rnd, parameters):
        # Call the original evaluate function
        loss, metrics = super().evaluate(rnd, parameters)
        self.current_accuracy = metrics.get("accuracy", 0.0)
        if self.current_accuracy >= self.target_accuracy:
            print(f"Target accuracy reached: {self.current_accuracy}")
            return None, metrics
        return loss, metrics

    def stop_condition(self):
        # Stop if the current accuracy meets or exceeds the target accuracy
        return self.current_accuracy >= self.target_accuracy

    def aggregate_fit(self, rnd, 
                      results: list[tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]], 
                      failures):
        # calculate communication cost
        communication_cost = 0
        for client_result in results:
            _, fit_res = client_result
            # Convert parameters to ndarrays
            weights = flwr.common.parameters_to_ndarrays(fit_res.parameters)
            weights_size = sum(w.nbytes for w in weights)
            communication_cost += weights_size
        
        self.total_communication_cost += communication_cost
        
        print(f"Round {rnd}: Total communication cost this round: {communication_cost / 1024} KB")
        print(f"Total communication cost so far: {self.total_communication_cost / 1024} KB")
        
        # Call the original aggregate_fit method
        return super().aggregate_fit(rnd, results, failures)

def server_fn_wrapper(target_accuracy):
    def server_fn(context: Context) -> ServerAppComponents:
        # Create the FedAvg strategy
        strategy = CommunicationTrackingStrategy(
            target_accuracy=target_accuracy,
            fraction_fit=1,
            fraction_evaluate=0.3,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=NUM_PARTITIONS,
            initial_parameters=ndarrays_to_parameters(params),
            evaluate_fn=evaluate,  # Pass the evaluation function
            on_fit_config_fn=fit_config,  # Pass the fit configuration function
        )

        # Configure the server for 10 rounds of training
        config = ServerConfig(num_rounds=10)
        return ServerAppComponents(strategy=strategy, config=config)
    
    return server_fn


def run_fl(args):

    global NUM_PARTITIONS
    NUM_PARTITIONS = args.num_partitions

    # Create the Client/ServerApp
    client = ClientApp(client_fn=client_fn_wrapper(iid=args.iid))
    server = ServerApp(server_fn=server_fn_wrapper(args.target_accuracy))

    # Specify the resources each of your clients need
    # By default, each client will be allocated 1x CPU and 0x GPUs
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

    # When running on GPU, assign an entire GPU for each client
    if DEVICE.type == "cuda":
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.1}}
        # Refer to our Flower framework documentation for more details about Flower simulations
        # and how to set up the `backend_config`

    def stopping_condition(context):
        return server.strategy.stop_condition()

    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_PARTITIONS,
        backend_config=backend_config,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_partitions", type=int, default=10, help="Number of clients")
    argparser.add_argument("--iid", type=bool, default=False, help="IID or non-IID partitioning")
    argparser.add_argument("--target_accuracy", type=float, default=0.9, help="Target accuracy to stop training")

    args = argparser.parse_args()

    run_fl(args)

