import torch
list1 = [torch.tensor(1.), torch.tensor(2.)]
list2 = [torch.tensor(3.), torch.tensor(4.)]
list3 = [list1,list2]
for item in list3:
    xx = [x.numpy() for x in item]
    print(xx)