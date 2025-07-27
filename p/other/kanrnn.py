
from rnn import RNN


model = RNN()


X = []
Y = [model.target_function(x) for x in X]
model.custom_backpropagation(X)
