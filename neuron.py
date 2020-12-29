# coding: utf8
# !/usr/bin/env python
import gzip, numpy, torch

ETA = 0.00001 # taux d'apprentissage

def sigmoid_activation(s):
    return 1 / (1 + torch.exp(-s))

def linear_activation(s):
    return s

class Layer:
    def __init__(self, number_of_neurons, activation_function):
        self.y = torch.empty((1, number_of_neurons), dtype=torch.float)
        self.b = torch.ones((1, number_of_neurons), dtype=torch.float)
        self.delta = torch.empty((1, number_of_neurons), dtype=torch.float)
        self.next_layer = None
        self.activation_function = activation_function

    def calculate_delta_error(self):
        yi = torch.mm(self.y, (1 - self.y.T))
        sum_delta_w = torch.mm(self.next_layer.delta, self.next_layer.w.T)
        self.delta = torch.mm(yi, sum_delta_w)

class EntryLayer(Layer):
    def __init__(self, number_of_neurons, activation_function, data_size):
        super(EntryLayer, self).__init__(number_of_neurons, activation_function)
        self.w = torch.empty((data_size, number_of_neurons), dtype=torch.float)
        torch.nn.init.uniform_(self.w, -0.001, 0.001)

    def activate(self, data):
        s = torch.mm(data, self.w) + self.b
        self.y = self.activation_function(s)

    def update_w(self, data):
        self.w += ETA * torch.mm(data.T, self.delta)

class MiddleLayer(Layer):
    def __init__(self, number_of_neurons, activation_function, previous_layer):
        super(MiddleLayer, self).__init__(number_of_neurons, activation_function)
        self.previous_layer = previous_layer
        self.w = torch.empty((previous_layer.y.shape[1], number_of_neurons), dtype=torch.float)
        torch.nn.init.uniform_(self.w, -0.001, 0.001)

    def activate(self):
        s = torch.mm(self.previous_layer.y, self.w) + self.b
        self.y = self.activation_function(s)

    def update_w(self):
        self.w += ETA * torch.mm(self.previous_layer.y.T, self.delta)

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