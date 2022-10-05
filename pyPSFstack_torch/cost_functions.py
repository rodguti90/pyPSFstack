import torch

def loss_loglikelihood(output, target):
    # loss = torch.mean(-target * torch.log(output) + output)
    loss = torch.mean(-target * torch.log(output/target) + output-target)
    return loss

def loss_sumsquared(output, target):
    loss = torch.mean( (output-target)**2 )
    return loss