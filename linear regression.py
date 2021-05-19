import numpy as np
import torch

# The training data can be represented using 2 matrices: inputs and targets, each with one row
# per observation and one column per variable
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')
# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

# We've separated the input and target variables because we'll operate on them separately.
# Also, we've created numpy arrays because this is typically how we'll work with training data:
# read some CSV files as numpy arrays, do some processing, and then convert them to pytorch tensors as follows:
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print("inputs:", inputs)
print("targets:", targets)

# Linear Regression Models from scratch
# The weights and biases (w11, w12, w13, w21, w22, w23, b1 and b2) can also be represented as matrices, initialized as
# random values . The first row of w and the first element of b are used to predict the first target
# variable i.e. yield of apples, and similarly the second for oranges.
w = torch.rand(2, 3, requires_grad=True)
b = torch.rand(2, requires_grad=True)
print("w:", w)
print("b:", b)


def model(x):
    # @ represents the matrix multiplication in PyTorch, and the .t method returns the transpose ofa tensor
    # The matrix obtained by passing the input data into the model is a set of predictions for the target variables
    return x @ w.t() + b


preds = model(inputs)
print("preds:", preds)
# Let's compare the predictions of our model to the actual targets
print("targets:", targets)


# There is a huge difference between the predictions of our model and the actual values of the target variables.
# Obviously, this is because we have initialized our model with random weights and biases, and we can't expect it
# to just work fine.

# Loss Function
# Before we improve our model, we need a way to evaluate how well our model is performing.
# We can compare the model's predictions with the actual targets, using the method called Mean Squared Error.
def mse(t1, t2):
    # torch.sum returns the sum of all the elements in a tensor,and the .numel method returns the number of
    # elements in a tensor.
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


# Compute loss
loss = mse(preds, targets)
print("loss:", loss)

# Compute gradients
loss.backward()
print("w.grad:", w.grad)
print("b.grad:", b.grad)

# The loss is a quadratic function of our weights and biases, and our objective is to find the set of weights
# where the loss is the lowest.

# Before we proceed, we need to reset the gradients to zero by calling .zero_()
# We need to do this because PyTorch accumulates gradients i.e. the next time we call .backward() on the loss,
# the new gradient values will be added to the existing values, which may lead to unexpected values.
w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)


