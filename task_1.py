import torch
import torch.nn as nn

class PowerModule(nn.Module):
    def __init__(self):
        super(PowerModule, self).__init__()
        #using clamp - is bad idea because it is constrain values
        #or sigmoid * 2 + 1

        # Initialize learnable parameter 'n' with a constraint in the range [1, 3]
        self.n = nn.Parameter(torch.rand(1) * 2 + 1)

    def forward(self, x):
        # Ensure n is within the range [1, 3]
        # also we can try use (sigmoid * 2 + 1) or other approach depending on the task
        self.n.data = torch.clamp(self.n, 1, 3)
        
        x = abs(x)
        y = torch.pow(x, self.n)
        
        return y
    