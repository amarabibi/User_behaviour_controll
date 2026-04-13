# li=(1,2,3,4,5,6,7,8,9,10)
# print(li)



import numpy  as num
import random as rand
# num.array={1,2,3,4,5,6}
# print(num.array)


print("Random number")

result = [num.random.rand(2,5)]
print(result)
import torch

print("Random 2D Tensor (Floats)")

result = torch.rand(3, 4)
print(result)

import torch
list1 = [1, 2, 3, 4, 5]
tensor1 = torch.tensor(list1)