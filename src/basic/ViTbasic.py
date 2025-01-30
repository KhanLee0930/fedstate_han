from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from PIL import Image

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

from transformers import ViTConfig, ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, Trainer

from utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS = 5
BATCH_SIZE = 32


def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Define transforms to resize and repeat channels
    image_processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    # Define a preprocessing function using the feature extractor
    def preprocess_function(examples):
        # Apply the feature extractor to the images
        # Apply the feature extractor to the images
        if isinstance(examples["image"], Image.Image):
            img = examples["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            examples["image"] = img

        processed_images = image_processor(images=examples["image"], return_tensors="pt")["pixel_values"]
        # Return a dictionary containing the processed images and labels
        return {"pixel_values": processed_images, "labels": examples["label"]}
    
    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.map(preprocess_function, remove_columns=["image", "label"])
    # trainloader = DataLoader(
    #     partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    # )
    # valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").map(preprocess_function, remove_columns=["image", "label"])
    # testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return partition_train_test["train"], partition_train_test["test"], testset


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

def get_trainer(model, training_args, train_dataset, eval_dataset):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset)

class BasicFlowerClient(NumPyClient):
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer

    def get_parameters(self, config):
        # Extract model parameters as NumPy arrays
        return get_parameters(self.model)

    def set_parameters(self, parameters, config):
        # Set model parameters from NumPy arrays
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        # Set model parameters
        self.set_parameters(parameters, config)
        
        # Train the model using Transformers' Trainer
        self.trainer.train()

        # Return updated parameters and number of examples
        return self.get_parameters(), len(self.trainer.train_dataset), {}

    def evaluate(self, parameters, config):
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate the model using Transformers' Trainer
        eval_result = self.trainer.evaluate()
        loss = eval_result["eval_loss"]

        return loss, len(self.trainer.eval_dataset), {"accuracy": eval_result.get("eval_accuracy", 0.0)}
    

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load pretrained model and feature extractor
    # Load the pretrained model configuration and update for MNIST
    config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
    config.num_labels = 10  # Update the number of labels for MNIST
    config.id2label = {str(i): str(i) for i in range(10)}  # Update id2label mapping
    config.label2id = {str(i): i for i in range(10)}  # Update label2id mapping
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', config=config, ignore_mismatched_sizes=True)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 10)

    # Load data (MNIST)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    train_dataset, val_dataset, test_dataset = load_datasets(partition_id)
    trainer = get_trainer(model, training_args, train_dataset, val_dataset)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return BasicFlowerClient(model, trainer).to_client()


# Create the ClientApp
client = ClientApp(client_fn=client_fn)


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=5)

    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=5,  # Never sample less than 5 clients for training
        min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
        min_available_clients=5,  # Wait until all 5 clients are available
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    return ServerAppComponents(strategy=strategy, config=config)


# Create the ServerApp
server = ServerApp(server_fn=server_fn)

# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign 1/5 GPU for each client
if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.2}}

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS,
    backend_config=backend_config,
)


