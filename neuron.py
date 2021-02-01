# coding: utf8
# !/usr/bin/env python
import gzip, torch

BATCH_SIZE = 5
ETA = 0.00001 # taux d'apprentissage

def sigmoid_activation(s):
    return 1 / (1 + torch.exp(-s))

def linear_activation(s):
    return s

class Layer:
    def __init__(self, number_of_neurons, activation_function):
        self.y = torch.empty((BATCH_SIZE, number_of_neurons), dtype=torch.float)
        self.b = torch.empty((1, number_of_neurons), dtype=torch.float)
        torch.nn.init.uniform_(self.b, -0.001, 0.001)
        self.delta = torch.empty((BATCH_SIZE, number_of_neurons), dtype=torch.float)
        self.next_layer = None
        self.activation_function = activation_function

    def calculate_delta_error(self):
        yi = torch.tensor([[e * (1 - e) for e in row] for row in self.y], dtype=torch.float)
        sum_delta_w = torch.mm(self.next_layer.w, self.next_layer.delta.T)
        self.delta = torch.multiply(yi, sum_delta_w.T)

class EntryLayer(Layer):
    def __init__(self, number_of_neurons, activation_function, data_size):
        super(EntryLayer, self).__init__(number_of_neurons, activation_function)
        self.w = torch.empty((data_size, number_of_neurons), dtype=torch.float)
        torch.nn.init.uniform_(self.w, -0.001, 0.001)

    def activate(self, data):
        s = torch.mm(data, self.w) + self.b
        self.y = self.activation_function(s)

    def update_wb(self, data):
        self.w += ETA * torch.mm(data.T, self.delta)
        self.b += ETA * self.delta.sum(axis=0)

class MiddleLayer(Layer):
    def __init__(self, number_of_neurons, activation_function, previous_layer):
        super(MiddleLayer, self).__init__(number_of_neurons, activation_function)
        self.previous_layer = previous_layer
        self.w = torch.empty((previous_layer.y.shape[1], number_of_neurons), dtype=torch.float)
        torch.nn.init.uniform_(self.w, -0.001, 0.001)

    def activate(self):
        s = torch.mm(self.previous_layer.y, self.w) + self.b
        self.y = self.activation_function(s)

    def update_wb(self):
        self.w += ETA * torch.mm(self.previous_layer.y.T, self.delta)
        self.b += ETA * self.delta.sum(axis=0)

class OutputLayer(MiddleLayer):
    def __init__(self, number_of_neurons, activation_function, previous_layer):
        super(OutputLayer, self).__init__(number_of_neurons, activation_function, previous_layer)

    def calculate_delta_error(self, t):
        self.delta = t - self.y

class SingleLayer(EntryLayer):
    def __init__(self, number_of_neurons, activation_function, data_size):
        super(SingleLayer, self).__init__(number_of_neurons, activation_function, data_size)

    def calculate_delta_error(self, t):
        self.delta = t - self.y