import torch

# `torch.classes.load_library()` allows you to pass the path to your .so file
# to load it in and make the custom C++ classes available to both Python and
# TorchScript
torch.classes.load_library("build/libmerlin_kv.so")
# You can query the loaded libraries like this:
print(torch.classes.loaded_libraries)
s = torch.classes.merlin_kv.HashTable()


# We can call methods in Python
s.init(8192, 8192, 1024 * 1024 * 1024 * 16, 0.75)
N = 1
DIM = 2
keys = torch.ones(N, dtype=torch.int64, device='cuda')
values = torch.ones(N * DIM, dtype=torch.float32, device='cuda')

found_values = torch.tensor(N * DIM, dtype=torch.float32, device='cuda')

found = torch.tensor([N], dtype=torch.bool, device='cuda')
s.insert_or_assign(N, keys, values)
s.find(N, keys, found_values, found)
print("found values: ", found_values, found)
print("s.size=", s.size())
s.clear()
print("after clear s.size=", s.size())
s.find(N, keys, found_values, found)
print("found values after clear: ", found_values, found)
