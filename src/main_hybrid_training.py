from __future__ import annotations

import warnings
import datasets.utils._dill


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
from utils.utils import parse_args, run_pre_experiments, Net, load_datasets, set_parameters, get_parameters, train, test
import StarlinkDataForFL.StarlinkData as Generator

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context


class FlowerClient(NumPyClient):
    def __init__(self, pid, net, local_iterations, lr, trainloader, valloader, device):
        self.pid = pid  # partition ID of a client
        self.net = net
        self.local_iterations = local_iterations
        self.lr = lr
        self.trainloader = trainloader
        self.valloader = valloader
        self.train_iter = iter(trainloader)
        self.device = device

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # Read values from config
        set_parameters(self.net, parameters)
        round_grad_norm_squared, loss, self.train_iter = train(self.net, self.train_iter, self.trainloader,
                                                               local_iterations=self.local_iterations, lr=self.lr,
                                                               device=self.device)
        metrics = {"round_grad_norm_squared": json.dumps(round_grad_norm_squared), "train_loss": json.dumps(loss)}
        return get_parameters(self.net), len(self.trainloader), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn_wrapper(save_dir: str, alpha: float, batch_size: int, local_iterations: int, lr: float, distribution,
                      device):
    def client_fn(context: Context) -> Client:
        net = Net().to(device)
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        print("partition_id",partition_id)
        print("num_partitions",num_partitions)
        trainloader, valloader, _, _ = load_datasets(save_dir, partition_id, num_partitions, batch_size, alpha=alpha,
                                                     distribution=distribution)
        return FlowerClient(partition_id, net, local_iterations, lr, trainloader, valloader, device).to_client()

    return client_fn


def create_evaluate_fn(save_dir: str, batch_size: int, p, device):
    _, _, testloader, _ = load_datasets(save_dir, 0, p, batch_size)

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
    def __init__(self, save_dir, batch_size, num_rounds, target_accuracy, **kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.num_rounds = num_rounds
        self.total_communication_cost = 0
        self.target_accuracy = target_accuracy
        self.current_accuracy = 0.0
        self.total_grad_norm_squared = []
        self.test_acc = []
        self.train_loss = []
        self.val_acc = []

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
        avg_grad_norm_squared = []
        train_loss = 0
        for client_result in results:
            _, fit_res = client_result
            # Convert parameters to ndarrays
            weights = flwr.common.parameters_to_ndarrays(fit_res.parameters)
            weights_size = sum(w.nbytes for w in weights)
            communication_cost += weights_size
            round_grad_norm_squared = json.loads(fit_res.metrics["round_grad_norm_squared"])
            avg_grad_norm_squared.append(round_grad_norm_squared)
            train_loss += json.loads(fit_res.metrics["train_loss"])
        self.total_grad_norm_squared.extend(
            [sum(grad_norm_squared[i] / len(avg_grad_norm_squared) for grad_norm_squared in avg_grad_norm_squared) for i
             in range(len(avg_grad_norm_squared[0]))])
        self.total_communication_cost += communication_cost
        self.test_acc.append(self.current_accuracy)
        self.train_loss.append(train_loss / len(results))
        # print(f"Round {rnd}: Total communication cost this round: {communication_cost / 1024} KB")
        # print(f"Total communication cost so far: {self.total_communication_cost / 1024} KB")
        if rnd == self.num_rounds:
            cumulative_average = np.cumsum(np.array(self.total_grad_norm_squared)) / np.arange(1,
                                                                                               len(self.total_grad_norm_squared) + 1)
            np.save(os.path.join(self.save_dir, "actual_GNS.npy"), cumulative_average)
            np.save(os.path.join(self.save_dir, "test_acc.npy"), np.array(self.test_acc))
            np.save(os.path.join(self.save_dir, "train_loss.npy"), np.array(self.train_loss))
        # Call the original aggregate_fit method
        return super().aggregate_fit(rnd, results, failures)

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[float, dict]], failures: List[BaseException]) -> \
    Optional[float]:
        val_acc = 0
        total_examples = 0
        for _, evaluate_res in results:
            total_examples += evaluate_res.num_examples
            val_acc += evaluate_res.metrics["accuracy"] * evaluate_res.num_examples
        self.val_acc.append(val_acc / total_examples)
        if server_round == self.num_rounds:
            np.save(os.path.join(self.save_dir, "val_acc.npy"), np.array(self.val_acc))
        return super().aggregate_evaluate(server_round, results, failures)

# save_dir, batch_size, target_accuracy, **kwargs
def server_fn_wrapper(save_dir, batch_size, num_rounds, target_accuracy, p, params, device):
    def server_fn(context: Context) -> ServerAppComponents:
        # Create the FedAvg strategy
        strategy = CommunicationTrackingStrategy(
            save_dir=save_dir,
            batch_size=batch_size,
            num_rounds=num_rounds,
            target_accuracy=target_accuracy,
            fraction_fit=1,
            fraction_evaluate=1,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=p,
            initial_parameters=ndarrays_to_parameters(params),
            evaluate_fn=create_evaluate_fn(save_dir, batch_size, p , device),  # Pass the evaluation function
            on_fit_config_fn=fit_config,  # Pass the fit configuration function
        )

        # Configure the server for 10 rounds of training
        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn


if __name__ == "__main__":
    np.random.seed(42)
    args = parse_args()
    save_dir = args.save_dir
    print(f"Save Directory: {save_dir}")

    # Initialize Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initial_model = Net().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    params = get_parameters(initial_model)
    if device.type == "cuda":
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1 / (args.p+1)}}
    else:
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

    # Lightweight Experiments
    configs, zeta, num_classes = run_pre_experiments(args.N, args.B, args.alpha, args.num_samples, initial_model,
                                                     criterion,
                                                     save_dir, device)
    print("Lightweight Experiments")
    print(configs, zeta, num_classes)
    print("Lightweight Experiments")

    # Critic
    critic = Critic(configs, zeta, args.E, args.epsilon, args.B, args.p, args.spi, device)

    # Satellite connectivity data
    data = Generator.Model_Generator(args.N, num_classes, critic.distribution.detach().cpu().numpy(), args.delta,
                                     args.p)
    Para = [data['N'], num_classes, data['K'], data['S']]
    print('Scheduler Starts')

    print("critic.distribution")
    print(critic.distribution)
    print("critic.distribution")


    original_distribution = torch.floor(critic.distribution).to(torch.int)
    keep = original_distribution - torch.floor(original_distribution * args.Pass2CenterRation).to(torch.int)
    sent = (original_distribution - keep).sum(dim=0)
    print("Num total sample before upload",keep.sum())
    actor = FedSateDataBalancer(keep, data['Psi'], data['Phi'], args.delta, data['C_access'],
                                data['C_e'], Para, data['S_set'], data['E_involved'], data['Client_Set'],
                                num_epochs=200001)

    if args.method == "Balance":
        _, distribution, _ = actor.balance_critic(critic)
    elif args.method == "IID":
        _, distribution, _ = actor.balance_iid()
    elif args.method == "Non-IID":
        _, distribution, _ = actor.balance_non_iid()
    elif args.method == "Random":
        _, distribution, _ = actor.balance_vanilla_uploading()

    distribution = torch.floor(distribution).to(torch.int)
    print("Num total sample after balancing", distribution.sum())
    distribution = torch.cat([distribution, sent.unsqueeze(0)], dim=0)
    print(args.method, distribution)
    # Obtain learning rate
    Lambda = critic.get_gradient_diversity(distribution)
    lr = critic.get_lr(Lambda)
    threshold = critic.get_threshold(lr)
    iteration_estimation, time_estimation = critic.time_estimation(distribution)
    print(
        f"Initial Loss: {critic.initial_loss}, L: {critic.L}, C1: {critic.C1}, sigma_squared: {critic.sigma_squared}, Lambda: {Lambda}, zeta: {zeta}, lr: {lr}, iteration estimation: {iteration_estimation}, threshold: {threshold}")
    convergence_prediction = critic.get_convergence_prediction(lr, args.num_rounds * args.E)
    np.save(os.path.join(save_dir, "predicted_GNS.npy"), convergence_prediction)
    # Create the Client/ServerApp
    client = ClientApp(client_fn=client_fn_wrapper(save_dir, args.alpha, args.B, args.E, lr, distribution, device))
    server = ServerApp(
        server_fn=server_fn_wrapper(save_dir, args.B, args.num_rounds, args.target_accuracy, args.p+1, params, device))
    # Run Simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=args.p+1,
        backend_config=backend_config,
    )
