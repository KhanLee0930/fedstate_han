from __future__ import annotations

import copy
import random
import warnings
import datasets.utils._dill
warnings.filterwarnings("ignore", category=UserWarning)
def suppress_dill_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="datasets.utils._dill")

suppress_dill_warnings()

from typing import Dict, Optional, Tuple

import os
import json
import torch
import numpy as np

from critic import Critic
from actor import FedSateDataBalancer
from utils.utils import parse_args, run_pre_experiments, Net, load_datasets, set_parameters, get_parameters, train, test, set_seed, load_datasets_distribution
import StarlinkDataForFL.StarlinkData as Generator

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
from huggingface_hub import HfApi

import sys

#token = "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"
#os.environ["HUGGINGFACE_TOKEN"] = token
#api = HfApi()
#user_info = api.whoami(token=token)
#print(user_info)

class FlowerClient(NumPyClient):
    def __init__(self, pid, net, local_iterations, lr, trainloader, valloader, weight_decay, device):
        self.pid = pid  # partition ID of a client
        self.net = net
        # We use simple SGD since that is what is specified in the paper. Also, Adam might cause divergence. Lastly, adding weight decay might prevent divergence
        self.optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
        self.local_iterations = local_iterations
        self.lr = lr
        self.trainloader = trainloader
        self.valloader = valloader
        self.train_iter = iter(trainloader)
        self.device = device

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        round_grad_norm_squared, losses, self.train_iter = train(self.net, self.optimizer, self.train_iter, self.trainloader,
                                                         local_iterations=self.local_iterations, device=self.device)
        metrics = {"round_grad_norm_squared": json.dumps(round_grad_norm_squared), "train_losses": json.dumps(losses)}
        return get_parameters(self.net), len(self.trainloader), metrics


def client_fn_wrapper(save_dir: str, batch_size: int, local_iterations: int, lr: float, distribution, weight_decay, device):
    def client_fn(context: Context) -> Client:
        net = Net().to(device)
        partition_id = context.node_config["partition-id"]
        trainloader, testloader, _ = load_datasets_distribution(save_dir, partition_id, batch_size, distribution)
        return FlowerClient(partition_id, net, local_iterations, lr, trainloader, testloader, weight_decay, device).to_client()

    return client_fn


def create_evaluate_fn(save_dir: str, batch_size: int, p, device):
    _, testloader, _ = load_datasets(save_dir, 0, p, batch_size)
    def evaluate(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        net = Net().to(device)
        set_parameters(net, parameters)  # Update model with the latest parameters
        loss, accuracy = test(net, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
    }
    return config

class CommunicationTrackingStrategy(FedAvg):
    def __init__(self, save_dir, batch_size, num_rounds, **kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.num_rounds = num_rounds
        self.current_accuracy = 0.0

        self.train_loss = []
        self.total_grad_norm_squared = []
        self.test_acc = []

    def evaluate(self, rnd, parameters):
        # Call the original evaluate function
        loss, metrics = super().evaluate(rnd, parameters)
        self.current_accuracy = metrics.get("accuracy", 0.0)
        return loss, metrics

    def aggregate_fit(self, rnd,
                      results: list[tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
                      failures):
        avg_grad_norm_squared = []
        train_loss = []
        # Aggregate results from all clients
        for client_result in results:
            _, fit_res = client_result
            avg_grad_norm_squared.append(json.loads(fit_res.metrics["round_grad_norm_squared"]))
            train_loss.append(json.loads(fit_res.metrics["train_losses"]))

        # Average results over clients
        self.total_grad_norm_squared.extend(np.mean(np.array(avg_grad_norm_squared), axis=0).tolist())
        self.train_loss.extend(np.mean(np.array(train_loss), axis=0).tolist())
        self.test_acc.append(self.current_accuracy)
        # Save results at the end of training
        if rnd == self.num_rounds:
            cumulative_average = np.cumsum(np.array(self.total_grad_norm_squared)) / np.arange(1,
                                                                                               len(self.total_grad_norm_squared) + 1)
            np.save(os.path.join(self.save_dir, "actual_GNS.npy"), cumulative_average)
            np.save(os.path.join(self.save_dir, "test_acc.npy"), np.array(self.test_acc))
            np.save(os.path.join(self.save_dir, "train_loss.npy"), np.array(self.train_loss))
        # Call the original aggregate_fit method
        return super().aggregate_fit(rnd, results, failures)


# save_dir, batch_size, target_accuracy, **kwargs
def server_fn_wrapper(save_dir, batch_size, num_rounds, p, params, device):
    def server_fn(context: Context) -> ServerAppComponents:
        # Create the FedAvg strategy
        strategy = CommunicationTrackingStrategy(
            save_dir=save_dir,
            batch_size=batch_size,
            num_rounds=num_rounds,
            fraction_fit=1,
            fraction_evaluate=1,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=p,
            initial_parameters=ndarrays_to_parameters(params),
            evaluate_fn=create_evaluate_fn(save_dir, batch_size, p, device),
            on_fit_config_fn=fit_config
        )

        # Configure the server for 10 rounds of training
        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn


if __name__ == "__main__":
    args = parse_args()

    print(f"running {args.method} with alpha {args.alpha} with seed {args.seed}")
    # Set random seed for numpy, torch, random, to ensure reproducibility
    set_seed(args.seed)

    # Define save directory, which is for the dataset, such that each experimental setup has its own directory to prevent overwriting
    save_dir = os.path.join(args.save_dir, f"{args.N}-{args.p}-{args.E}-{args.alpha}-{args.B}", f"data_shairng-{args.balance_params}-{args.seed}")
    print(f"Save Directory: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # Initialize Device, Model, Criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initial_model = Net().to(device)
    params = get_parameters(initial_model)
    criterion = torch.nn.CrossEntropyLoss()
    if device.type == "cuda":
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1 / (args.p//2 + 1)}}
    else:
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

    # Lightweight Experiments, assuming we run it on IID dataset (alpha=-1)
    configs, num_classes, model_size, spi_predictor = run_pre_experiments(args.N, args.B, args.num_samples, copy.deepcopy(initial_model), criterion, save_dir, device)
    print("Lightweight Experiments Output:")
    print(f"L: {configs[0]}, C1: {configs[1]}, sigmasquared: {configs[2]}, Loss: {configs[3]}, Zeta: {configs[4]}, SPI: {spi_predictor.predict(torch.tensor(args.B))}, Model Size: {model_size}")

    # Satellite Connectivity Data
    data, initial_distribution = Generator.Model_Generator(args.N, args.alpha, save_dir)
    print("Output for Satellite Connectivity Data")
    print(f"Average visible satellites by each user: {np.mean(np.sum(data['Psi'], axis=1))}, "
          f"Average number of satellites between two satellites: {np.mean(np.sum(data['Phi'], axis=1))}, "
          f"Uploading speed: {np.mean(data['C_access'])}, Transmission speed: {np.mean(data['C_e'])}, "
          f"Number of users: {data['N']}, Number of classes: {num_classes}, Number of clients: {args.p}, "
          f"Number of satellites: {data['S']}, Average visible satellites by each satellite: {np.mean(np.sum(data['G'], axis=1))}")

    # Define Actor
    actor = FedSateDataBalancer(initial_distribution, data['Psi'], data['Phi'], args.delta, data['C_access'], data['C_e'], data['N'], num_classes, args.p, data['S'], data['S_set'], data['E_involved'], num_epochs=args.num_actor_rounds, model_size=model_size, device=device)

    # Define Critic
    critic = Critic(configs, args.E, args.epsilon, args.B, args.p, spi_predictor, actor.communication_time)

    print('Scheduler Starts')
    _, distribution, _, batch_size, E = actor.uploading_strategy(args.method, critic, args.balance_params, args.B, args.E)

    distribution = torch.floor(distribution).to(torch.int)
    # This is for data sharing part
    indices = torch.randperm(args.p)
    sum_indices = indices[:args.p//2]
    remain_indices = indices[args.p//2:]
    sum_part = distribution[sum_indices]
    remain_part = distribution[remain_indices]
    sum_part = sum_part.sum(dim=0, keepdim=True)
    distribution = torch.cat([sum_part, remain_part], dim=0)
    critic.p = args.p//2 + 1
    print(f"New distribution is with dim {distribution.shape}")
    print(f"Final distribution obtained for {args.method}: {distribution}")
    print(f"Predict initial lambda: {critic.get_gradient_diversity(initial_distribution.to(device))}, predicted final lambda: {critic.get_gradient_diversity(distribution.to(device))}")
    # Using the final distribution, we obtain the parameters of the training
    Lambda = critic.get_gradient_diversity(distribution)
    lr = critic.get_lr(Lambda,batch_size, E)

    # Save the estimation obtained by the critic
    GNS_prediction = critic.get_GNS_prediction(lr, args.num_rounds * args.E)
    np.save(os.path.join(save_dir, "GNS_prediction.npy"), GNS_prediction)

    # Create the Client/ServerApp
    client = ClientApp(client_fn=client_fn_wrapper(save_dir, batch_size, E, lr, distribution, args.weight_decay, device))
    server = ServerApp(
        server_fn=server_fn_wrapper(save_dir, batch_size, args.num_rounds, args.p//2 + 1, params, device))

    # Run Simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=args.p//2 + 1,
        backend_config=backend_config,
    )