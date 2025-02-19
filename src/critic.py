import torch
import numpy as np

class Critic(object):
    def __init__(self, configs, E, epsilon, batch_size, p, spi_predictor, communication_time):
        self.L = float(configs[0])
        self.C1 = float(configs[1])
        self.C1_times_B = self.C1 * batch_size
        self.sigma_squared = float(configs[2])
        self.initial_loss = float(configs[3])
        self.zeta = float(configs[4])
        self.class_gradients = configs[5] # (num_classes, d)

        self.E = int(E)
        self.epsilon = float(epsilon)
        self.B = int(batch_size)
        self.p = int(p)
        self.spi_predictor = spi_predictor
        self.communication_time = communication_time

    # Takes in distribution as a parameter from Actor to estimate the time taken for training
    def time_estimation(self, distribution, batch_size, E):
        # Get gradient diversity Lambda
        Lambda = self.get_gradient_diversity(distribution)
        # Use Lambda to predict learning rate
        lr = self.get_lr(Lambda, batch_size, E)
        # Use learning rate to predict number of iterations required for convergence
        convergence_iterations = self.get_convergence_rounds(lr)
        # Use number of iterations and seconds per iteration to predict convergence time
        convergence_time = convergence_iterations * self.spi_predictor.predict(batch_size)
        # Number of FL rounds required to obtain the cost of communication
        FL_rounds = convergence_iterations / E
        total_communication_time = FL_rounds * self.communication_time
        return convergence_iterations, convergence_time + total_communication_time, Lambda, lr

    def get_gradient_diversity(self, distribution):
        q_values = torch.sum(distribution, dim=1) / torch.sum(distribution) # (num_partitions), relative weight of each partition

        # For each partition, obtain a distribution over the classes
        distribution_fractions = distribution / torch.sum(distribution, dim=1, keepdim=True) # (num_partitions, num_classes)

        # Average gradient for each partition
        partition_gradients = torch.matmul(distribution_fractions, self.class_gradients) # (num_partitions, d)

        # Gradient diversity based on formula
        sum_norm = sum([q_values[i] * torch.norm(partition_gradients[i])**2 for i in range(self.p)])
        norm_sum = torch.norm(sum([q_values[i] * partition_gradients[i] for i in range(self.p)]))**2
        Lambda = sum_norm / norm_sum
        return Lambda

    # Uses the formula to obtain learning rate
    def get_lr(self, Lambda, batch_size, E):
        C1 = self.C1_times_B / batch_size
        # Constant of quadratic term
        a_1 = 2*(self.L**2)*C1*E / (1-(self.zeta**2))
        a_2 = (self.L**2)*(E**2)/(1-self.zeta)
        a_3 = 2 * (self.zeta**2) / (1+self.zeta)
        a_4 = 2 * self.zeta / (1-self.zeta)
        a_5 = (E - 1) / E
        a = a_1 + a_2 * (a_3 + a_4 + a_5)
        # Constant of linear term
        b = self.L * (C1/self.p + 1)

        # Constant term
        c = -1 * 1/Lambda

        discriminant = b**2 - 4 * a * c
        sqrt_discriminant = torch.sqrt(discriminant)
        eta2 = (-b + sqrt_discriminant) / (2 * a)
        # Return the larger learning rate
        return eta2

    def get_convergence_rounds(self, eta):
        return 1 / (eta * self.epsilon)

    def get_GNS_prediction(self, eta, num_iterations):
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