from kan import *


class RNN:

    def __init__(self):
        self.model = KAN()

    def target_function(self, x):
        return x * (x - 1)

    def custom_backpropagation(self, data):
        pass
