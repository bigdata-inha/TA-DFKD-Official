import torch

'''
Implementation of the forward hook to track feature statistics and compute a loss on them.

We will compute mean and variance, and will use l2 as a loss
'''

class DeepInversionFeatureHook():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.mean = 0
        self.nch = 0
        self.var = 0
        self.feature =0
        self.input = None
    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization

        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)


        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.running_mean = module.running_mean.data
        self.running_var = module.running_var.data

        self.mean = mean
        self.nch = nch
        self.var = var
        self.feature = r_feature
        self.input = input

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


class DeepInversionFeatureHook_check():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.running_mean = None
        self.running_var = None
    def hook_fn(self, module, input, output):
        self.running_mean = torch.mean(module.running_mean.data)
        self.running_var = torch.mean(module.running_var.data)
    def close(self):
        self.hook.remove()
