import torch

t = torch.rand(4, 768)
# print(t)
# idx = torch.randperm(t.shape[0])
# print(idx)
# t = t[idx].view(t.size())
#
# print(t[2:4])

print(torch.randn(20))
print(torch.randint(20, size=(20,)))
print(torch.rand(20))
print(torch.rand_like(t))
print(torch.randn_like(t))
print(torch.randint_like(t, high=20))
print(torch.randperm(20))
