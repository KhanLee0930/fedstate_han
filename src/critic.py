import torch
import numpy as np

class TrainingCritic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, distribution, critic):
        # Compute the scalar output we want to differentiate
        # Here, we choose the predicted convergence_time as our differentiable scalar.
        _, convergence_time = critic.time_estimation(distribution)

        # Save for backward
        ctx.save_for_backward(distribution)
        ctx.critic = critic

        return convergence_time

    @staticmethod
    def backward(ctx, grad_output):
        (distribution,) = ctx.saved_tensors
        critic = ctx.critic

        # Use torch.autograd.grad to compute gradients
        grad_distribution = critic.class_gradients.clone().detach()

        # The critic itself is not a tensor, so it doesn't need a gradient
        return grad_distribution, None


class Critic(object):
    def __init__(self, configs, zeta, E, epsilon, batch_size, p, spi, device):
        self.zeta = float(zeta)
        self.E = int(E)
        self.epsilon = float(epsilon)
        self.B = int(batch_size)
        self.p = int(p)
        self.spi = float(spi)

        self.L = float(max([config[0] for config in configs]))
        self.C1 = float(max([config[1] for config in configs]))
        self.sigma_squared = float(max([config[2] for config in configs]))
        self.initial_loss = float(sum([config[3] for config in configs]) / len(configs))
        # Detach it to make it a leaf tensor so the .grad attribute will be filled
        distribution_data = torch.stack([config[5] for config in configs])  # Create the data distribution on ground users
        self.distribution = distribution_data.clone().detach().to(dtype=torch.float32, device=device)

        torch.cuda.empty_cache()

        with torch.no_grad():
            gradients = torch.stack([config[4] for config in configs]) # (num_partitions, num_classes, d)
            self.class_gradients = (torch.sum(gradients * torch.unsqueeze(self.distribution, dim=-1), dim=0) / torch.unsqueeze(torch.sum(self.distribution, dim=0), dim=-1)).clone().detach() # (num_classes, d)

    # Instead of using self.distribution, takes in distribution as a parameter from Actor
    def time_estimation(self, distribution):
        Lambda = self.get_gradient_diversity(distribution)
        lr = self.get_lr(Lambda)
        convergence_rounds = self.get_convergence_rounds(lr)
        convergence_iterations = convergence_rounds * self.E
        convergence_time = convergence_iterations * self.spi
        return convergence_iterations, convergence_time

    def get_gradient_diversity(self, distribution):
        q_values = torch.sum(distribution, dim=1) / torch.sum(distribution) # (num_partitions), relative weight of each partition

        distribution_fractions = distribution / torch.sum(distribution, dim=1, keepdim=True) # (num_partitions, num_classes)

        partition_gradients = torch.matmul(distribution_fractions, self.class_gradients) # (num_partitions, d)
        sum_norm = sum([q_values[i] * torch.norm(partition_gradients[i])**2 for i in range(self.p)])

        norm_sum = torch.norm(sum([q_values[i] * partition_gradients[i] for i in range(self.p)]))**2

        # Compute gradient diversity
        Lambda = sum_norm / norm_sum
        return Lambda

    def get_lr(self, Lambda):
        # Constant of quadratic term
        a_1 = 2*(self.L**2)*self.C1*self.E / (1-(self.zeta**2))
        a_2 = (self.L**2)*(self.E**2)/(1-self.zeta)
        a_3 = 2 * (self.zeta**2) / (1+self.zeta)
        a_4 = 2 * self.zeta / (1-self.zeta)
        a_5 = (self.E - 1) / self.E
        a = a_1 + a_2 * (a_3 + a_4 + a_5)
        # Constant of linear term
        b = self.L * (self.C1/self.p + 1)

        # Constant term
        c = -1 * 1/Lambda

        discriminant = b**2 - 4 * a * c
        sqrt_discriminant = torch.sqrt(discriminant)
        eta2 = (-b + sqrt_discriminant) / (2 * a)
        # Return the larger learning rate
        return eta2
    
    def get_convergence_rounds(self, eta):
        # a = 2*(self.initial_loss) / eta
        # b = (self.L*self.sigma_squared)/(self.p*self.B)
        # c = (self.L**2 * eta**2 * self.sigma_squared) / self.B
        # d = (1+self.zeta**2) / (1-self.zeta**2) * self.E - 1
        # return a / (self.epsilon - (b + c * d))
        return 1 / (eta * self.epsilon)

    def get_convergence_prediction(self, eta, num_iterations):
        eta = eta.detach().cpu().numpy()
        T = np.arange(num_iterations) + 1
        a = 2*(self.initial_loss) / (eta * T)
        b = (eta*self.L*self.sigma_squared)/(self.p*self.B)
        c = (self.L**2 * eta**2 * self.sigma_squared) / self.B
        d = (1+self.zeta**2) / (1-self.zeta**2) * self.E - 1
        return a + b + c * d

    def get_threshold(self, eta):
        a = 2*(self.initial_loss) * self.epsilon
        b = (eta*self.L*self.sigma_squared)/(self.p*self.B)
        c = (self.L**2 * eta**2 * self.sigma_squared) / self.B
        d = (1+self.zeta**2) / (1-self.zeta**2) * self.E - 1
        return a + b + c * d

if __name__ == '__main__':
    from torch.autograd import gradcheck

    input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True))
    test = gradcheck(TrainingCritic.apply, input, eps=1e-6, atol=1e-4)
    print(test)
