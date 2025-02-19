import torch
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
import random
import copy
import time
from torch.utils.data import DataLoader

class SPI:
    def __init__(self, degree):
        self.degree = degree
    def fit(self, x, y):
        self.coefficients = np.polyfit(np.log(np.array(x)), np.log(np.array(y)), self.degree)
        self.coefficients = [torch.tensor(c, dtype=torch.float32, requires_grad=False) for c in self.coefficients]
    def predict(self, x):
        x = torch.log(x)
        pred = sum(c * x**i for i, c in enumerate(reversed(self.coefficients)))
        return torch.exp(pred)

class PreConfig(object):

    def __init__(self, Net, criterion, dataset, num_samples, num_classes, device):
        self.Net = Net
        self.criterion = criterion
        self.device = device
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.train_loader = dataset

    def update_model(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for batch in self.train_loader:
            optimizer.zero_grad()
            inputs, targets = batch["img"].to(self.device), batch["label"].to(self.device)
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()  # Accumulate the gradients across the batches
            optimizer.step()
        model.zero_grad()

    def reinit_model(self):
        return self.Net().to(self.device)

    # Monte Carlo estimation of L
    def get_L(self, epsilon=1e-3):
        model = self.reinit_model()
        L_values = []
        for _ in range(self.num_samples):
            # Sample two random weights
            u = {name: param.data.clone() for name, param in model.named_parameters()}
            # Apply perturbation to the weights to get v
            v = {name: param.data + epsilon * torch.randn_like(param) for name, param in model.named_parameters()}
            param_diff_norm = torch.sqrt(sum((u[name] - v[name]).norm() ** 2 for name in u))

            # Get Grad Norm for u
            model.zero_grad()
            model.load_state_dict(u, strict=False)
            for batch in self.train_loader:
                inputs, targets = batch["img"].to(self.device), batch["label"].to(self.device)
                # Compute gradients at u
                outputs_u = model(inputs)
                loss_u = self.criterion(outputs_u, targets) / len(self.train_loader)
                loss_u.backward() # Accumulate the gradients across the batches
            grad_u = [param.grad.clone() for param in model.parameters() if param.requires_grad]

            # Get Grad Norm for v
            model.zero_grad()
            model.load_state_dict(v, strict=False)
            for batch in self.train_loader:
                inputs, targets = batch["img"].to(self.device), batch["label"].to(self.device)
                # Compute gradients at u
                outputs_u = model(inputs)
                loss_v = self.criterion(outputs_u, targets) / len(self.train_loader)
                loss_v.backward() # Accumulate the gradients across the batches
            grad_v = [param.grad.clone() for param in model.parameters() if param.requires_grad]
            model.zero_grad()

            grad_diff_norm = torch.sqrt(sum((gu - gv).norm() ** 2 for gu, gv in zip(grad_u, grad_v)))

            # Compute L estimate for this sample
            L_sample = grad_diff_norm / (
                    param_diff_norm + 1e-8)  # Add a small constant to avoid division by zero
            L_values.append(L_sample.item())

            model = self.reinit_model()

        # Estimate L as the max of sampled values
        L_estimate = np.max(L_values)
        self.L = L_estimate
        return L_estimate

    # Calculate C1 and sigma
    def get_C1_sigma(self, model, B):
        dataset = self.train_loader.dataset  # Extract the dataset
        batch_size = self.train_loader.batch_size  # Mini-batch size

        batch_variances = []
        grad_norm_squares = []

        for _ in range(self.num_samples):
            # Reset Model
            model.zero_grad()
            # Grad Norm Squares
            for batch in self.train_loader:
                inputs, targets = batch["img"].to(self.device), batch["label"].to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets) / len(self.train_loader)
                loss.backward()
            full_grad = [param.grad.clone() for param in model.parameters() if param.requires_grad]
            full_grad_norm_sq = sum(g.norm()**2 for g in full_grad).item()
            grad_norm_squares.append(full_grad_norm_sq)
            model.zero_grad()

            # Batch Variance
            batch_variance = 0
            for _ in range(200):
                indices = random.sample(range(len(dataset)), batch_size)  # Random indices
                batch = [dataset[i] for i in indices]  # Randomly selected mini-batch
                inputs = torch.stack([b["img"] for b in batch]).to(self.device)
                targets = torch.as_tensor([b["label"] for b in batch]).to(self.device)


                # Compute mini-batch gradient
                model.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                mini_grad = [param.grad.clone() for param in model.parameters() if param.requires_grad]
                batch_variance += sum((mg - fg).norm()**2 for mg, fg in zip(mini_grad, full_grad)).item() / len(self.train_loader)
            batch_variances.append(batch_variance)
            model.zero_grad()
            self.update_model(model)

        # Solve for C1 and sigma^2 / B
        batch_variances = np.array(batch_variances)
        grad_norm_squares = np.array(grad_norm_squares)
        grad_norm_squares = sm.add_constant(grad_norm_squares)
        # print("batch_variances shape:", batch_variances.shape)
        # print("grad_norm_squares shape:", grad_norm_squares.shape)
        model = sm.QuantReg(batch_variances, grad_norm_squares)
        result = model.fit(q=0.95)
        sigma_squared_over_B, C1 = result.params
        sigma_squared = sigma_squared_over_B * B
        self.C1, self.sigma_squared = C1, sigma_squared
        return C1, sigma_squared

    def get_loss(self, model):
        # The inital loss should be based on the fixed base model
        total_loss = 0
        model.zero_grad()
        for batch in self.train_loader:
            inputs, targets = batch["img"].to(self.device), batch["label"].to(self.device)

            # Forward pass to compute the loss
            outputs = model(inputs)
            loss = self.criterion(outputs, targets) / len(self.train_loader)
            total_loss += loss.item()
        model.zero_grad()
        self.loss = total_loss
        return total_loss

    def get_gradient(self, model):
        model.zero_grad()
        accumulated_gradients = {cls: None for cls in range(self.num_classes)} # Representative gradient for each class
        class_counts = {cls: 0 for cls in range(self.num_classes)}
        for batch in self.train_loader:
            inputs, targets = batch["img"].to(self.device), batch["label"].to(self.device)
            for cls in range(self.num_classes):
                mask = (targets == cls)
                if mask.sum() == 0:  # Skip if no samples belong to this class
                    continue
                inputs_class = inputs[mask]
                targets_class = targets[mask]
                outputs = model(inputs_class)
                loss = self.criterion(outputs, targets_class) * mask.sum() # Equivalent to summing of losses
                model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    gradients = torch.flatten(torch.cat([param.grad.clone().flatten() for param in model.parameters() if param.grad is not None]))
                    if accumulated_gradients[cls] is None:
                        accumulated_gradients[cls] = gradients
                    else:
                        accumulated_gradients[cls] += gradients
                    class_counts[cls] += mask.sum().item()

        self.average_gradients = torch.stack([accumulated_gradients[cls] / class_counts[cls] if class_counts[cls] != 0
                                            else torch.zeros_like(gradients) for cls in range(self.num_classes)]) # (num_classes, d)
        self.class_counts = torch.tensor([class_counts[cls] for cls in range(self.num_classes)]) # (num_classes)
        return self.average_gradients, self.class_counts


    def get_spi(self, model):
        num_samples = 10
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
        spi = []
        for batch_size in batch_sizes:
            train_loader = DataLoader(self.train_loader.dataset, batch_size=batch_size)
            start = time.time()
            for idx, batch in enumerate(train_loader):
                if idx == num_samples:
                    break
                # Simulate forward and backwards passes
                inputs, targets = batch["img"].to(self.device), batch["label"].to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                model.zero_grad()
            seconds_per_iteration = (time.time() - start)/num_samples
            spi.append(seconds_per_iteration)
        spi_predictor = SPI(degree=3)
        spi_predictor.fit(batch_sizes, spi)
        return spi_predictor