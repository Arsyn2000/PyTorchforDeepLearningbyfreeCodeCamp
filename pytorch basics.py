# PYTORCH BASICS
# Begin by importing torch. PyTorch is a library for processing the tensors. A tensor is a number, vector, matrix,
# or any n-dimensional array.
import torch

# Let's create a tensor with a single number
# 4. is the shorthand of 4.0 It is used to indicate Python(and PyTorch) that you want to create
# a floating point number. We can verify this be checking the dtype attribute of our tensor.
t1 = torch.tensor(4.)
print(t1.dtype)

# Vector
# 1. is applied here. This will convert the other members of the list to float.
# Each element in the tensor should have the same datatype. A tensor has a regular shape unlike list.
t2 = torch.tensor([1., 2, 3, 4])
print(t2)

# Matrix
t3 = torch.tensor([[5., 6], [7, 8], [9, 10]])
print(t3)

# 3-D Array
t4 = torch.tensor([
    [
        [11, 12, 13],
        [13, 14, 15]
    ],
    [
        [15, 16, 17],
        [17, 18, 19.]
    ]
])
print(t4)

# Tensors can have any number of dimensions, and different lengths along each dimension. We can inspect
# the length along each dimension using the .shape property of tensor.
print("t1.shape:", t1.shape)
print("t2.shape:", t2.shape)
print("t3.shape:", t3.shape)
print("t4.shape:", t4.shape)

# Tensor operations and gradients
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)
# Arithmetic operations
y = w * x + b
print(y)
# Compute derivatives
print(y.backward())
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad)

# Interoperability with Numpy
# Numpy is a popular open source library used for mathematical and scientific computing in Python.
# It enables efficient opeartions on large multi-dimensional arrays, and has a large ecosystem of
# supporting libraries:
# Matplotlib for plotting and visualization
# OpenCV for image and video processing
# Pandas for file I/O and data analysis
import numpy as np

x = np.array([[1, 2], [3, 4.]])
print(x)

# We can convert a numpy array to a PyTorch tensor using torch.from_numpy
y = torch.from_numpy(x)
print(y)
z = torch.tensor(x)
print(z)
# The difference between the above two approaches: from_numpy uses the same space while the .tensor
# creates a copy and hence occupies extra space
print(x.dtype)
print(y.dtype)
print(z.dtype)
# Converting  a PyTorch tensor to a Numpy array using .numpy method of tensor
w = y.numpy()
print(w)
print(w.dtype)

# The interoperability is important because most datasets you'll work with will likely be read and preprocessed
# as Numpy arrays.

