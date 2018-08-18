import utils

import torch


class ModelPruner:
    def __init__(self, model, config):
        """
        Initialize model pruner.
        """
        self.weights_count = utils.count_parameters(model)
        self.pruners = []
        for name, weight in model.named_parameters():
            if 'bias' not in name:
                self.pruners.append(WeightPruner(weight, config[name]))

    def step(self, current_itr):
        """
        Used after each training step. Applies mask to every weight
        Also updates mask at regular intervals with self.freq frequency
        (different for each layer)
        """
        for pruner in self.pruners:
            pruner.apply_mask()
            pruner.update_mask(current_itr)

    def log(self):
        """
        Return total model sparsity
        """
        count_pruned = sum(pruner.get_pruned_count() for pruner in self.pruners)
        return count_pruned / self.weights_count


class WeightPruner:
    def __init__(self, weight, config):
        self.mask = torch.ones_like(weight, requires_grad=False)
        self.weight = weight
        start_itr = config['start_itr']
        ramp_itr = config['ramp_itr']
        end_itr = config['end_itr']
        freq = config['freq']
        q = config['q']

        # calculate the start slope and get
        # ramp slope by multiplying it
        self.start_slope = (2 * q * freq) / (2 * (ramp_itr - start_itr) + 3 * (end_itr - ramp_itr))
        self.ramp_slope = self.start_slope * config['ramp_slope_mult']
        self.start_itr = start_itr
        self.ramp_itr = ramp_itr
        self.end_itr = end_itr
        self.freq = freq

    def update_mask(self, current_itr):
        if current_itr > self.start_itr and current_itr < self.end_itr:
            if current_itr % self.freq == 0:
                th = self.start_slope * (current_itr - self.start_itr + 1) / self.freq
            else:
                th = (self.start_slope * (self.ramp_itr - self.start_itr + 1) +
                      self.ramp_slope * (current_itr - self.ramp_itr + 1)) / self.freq
            self.mask = torch.gt(torch.abs(self.weight.data), th).type(self.weight.type())

    def apply_mask(self):
        self.weight.data = self.weight.data * self.mask

    def get_weights_count(self):
        return int(torch.sum(self.mask).item())

    def get_pruned_count(self):
        return self.mask.numel() - self.get_weights_count()

    def get_density(self):
        return self.get_weights_count() / self.mask.numel()

    def get_sparsity(self):
        return self.get_pruned_count() / self.mask.numel()
