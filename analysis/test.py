from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
from PIL import Image

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS = 5
BATCH_SIZE = 32


def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Define transforms to resize and repeat channels
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    # Define a preprocessing function using the feature extractor
    def preprocess_function(examples):
        # Apply the feature extractor to the images
        if isinstance(examples["image"], Image.Image):
            img = examples["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            examples["image"] = img

        processed_images = feature_extractor(images=examples["image"], return_tensors="pt")["pixel_values"]
        # Return a dictionary containing the processed images and labels
        return {"pixel_values": processed_images, "labels": examples["label"]}
    
    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.map(preprocess_function)
    print(partition_train_test)
    # trainloader = DataLoader(
    #     partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    # )
    # valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").map(preprocess_function)
    # testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return partition_train_test["train"], partition_train_test["test"], testset


load_datasets(0)