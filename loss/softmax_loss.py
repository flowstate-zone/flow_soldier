import torch
import torch.nn as nn
from torch.nn import functional as F
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # def forward(self, inputs, targets):
    #     """
    #     Args:
    #         inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
    #         targets: ground truth labels with shape (num_classes)
    #     """
    #     print(inputs.shape)
    #     log_probs = self.logsoftmax(inputs)
    #     targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
    #     if self.use_gpu: targets = targets.cuda()
    #     targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    #     loss = (- targets * log_probs).mean(0).sum()
    #     return loss
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        # get the unique target
        unique_targets = torch.unique(targets)
        # create a dictionary to map unique targets to indices
        unique_targets = sorted(unique_targets)
        target_to_index = {target.item(): i for i, target in enumerate(unique_targets)}
        # print('target_to_index:', target_to_index)

        # print('inputs:', inputs)

        # update inputs and targets
        targets = torch.tensor([target_to_index[target.item()] for target in targets])
        inputs = inputs[:, unique_targets]

        # print(inputs.shape)

        log_probs = self.logsoftmax(inputs)
        zeros = torch.zeros(log_probs.size())
        
        # print('before')
        # import pdb; pdb.set_trace()
        targets = zeros.scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        # print('after')
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        return (-targets * log_probs).mean(0).sum()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()