import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import numpy as np
import itertools
import csv
import os
import shutil


class Identity(nn.Module):
    """Used for extracting features before fc layer
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ClassLabel(object):
    def __init__(self, label_file):
        """Initialize target transform

        :param label_file: the saved .npy file which contains labels of all samples
        """
        self.labels = np.load(label_file)

    def __call__(self, index):
        return self.labels[index]


class TransformTwice(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)

        return out1, out2


class TwoStreamSampler(Sampler):
    """A Sampler used for dataloader's argument 'batchsampler', It samples two subsamples
    coming from different sources and ensemble them in a single batch 

    Args:
        primary_indices (sequence): list of sample indices of a data source
        secondary_indices (sequence): list of sample indices of another data source
        batch_size (int): desired batch size
        secondary_batch_size (int): number of samples coming from secondary_indices
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.primary_batch_size = batch_size - secondary_batch_size
        self.secondary_batch_size = secondary_batch_size

        # check that given indice list length is valid
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        """method to iteratively return a batch, since the majority
        comes from primary indices, so apply iterate_eternally on secondary_indices"""
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        # we see here in a batch the samples from primary indices always comes first
        return (primary_batch + secondary_batch for (primary_batch, secondary_batch)
                in zip(grouper(primary_iter, self.primary_batch_size),
                       grouper(secondary_iter, self.secondary_batch_size)))

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


class ThreeStreamSampler(Sampler):
    """A Sampler used for dataloader's argument 'batchsampler'. It samples 3 subsamples coming
    from different sources and ensemble them in a single batch

    Args:
        primary_indices (sequence): list of sample indices of a data source
        secondary_indices (sequence): list of sample indices of another data source
        batch_size (int): desired batch size
        secondary_batch_size (int): number of samples coming from secondary_indices
    """
    def __init__(self, primary_indices, secondary_indices, primary_batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.primary_batch_size = primary_batch_size
        self.secondary_batch_size = secondary_batch_size

        # check that given indice list length is valid
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        """method to iteratively return a batch, since the majority
        comes from primary indices, so apply iterate_eternally on secondary_indices"""
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        # we see here in a batch the samples from primary indices always comes first
        return (primary_batch + primary_batch + secondary_batch for (primary_batch, secondary_batch)
                in zip(grouper(primary_iter, self.primary_batch_size),
                       grouper(secondary_iter, self.secondary_batch_size)))

    # drop last is true for primary indices
    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


class FourStreamSampler(Sampler):
    """A Sampler used for dataloader's argument 'batchsampler'. It samples 4 subsamples coming
    from different sources and ensemble them in a single batch

    Args:
        primary_indices (sequence): list of sample indices of a data source
        secondary_indices (sequence): list of sample indices of another data source
        batch_size (int): desired batch size
        secondary_batch_size (int): number of samples coming from secondary_indices
    """
    def __init__(self, primary_indices, secondary_indices, primary_batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.primary_batch_size = primary_batch_size
        self.secondary_batch_size = secondary_batch_size

        # check that given indice list length is valid
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        """method to iteratively return a batch, since the majority
        comes from primary indices, so apply iterate_eternally on secondary_indices"""
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        # we see here in a batch the samples from primary indices always comes first
        return (primary_batch + primary_batch + secondary_batch + secondary_batch for (primary_batch, secondary_batch)
                in zip(grouper(primary_iter, self.primary_batch_size),
                       grouper(secondary_iter, self.secondary_batch_size)))

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """Collect data into a fixed-length chunks

    :param iterable:
    :param n:
    :return:
    """
    args = [iter(iterable)] * n

    return zip(*args)


class AverageMeter(object):
    """Compute and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def combined_criterion(logits, targets, opt):
    """Calculate softmax loss but return the seperate loss and the final loss"""
    assert logits.is_cuda and targets.is_cuda
    batch_size = logits.size(0)
    # the way for this need to be consistent with the way you feed TwoStreamSampler
    label_length = int(batch_size * opt.ratio)
    unlabel_length = batch_size - label_length
    # unlabel comes first
    unlabel_logits = logits[:unlabel_length, :]
    unlabel_targets = targets[:unlabel_length]
    label_logits = logits[unlabel_length:, :]
    label_targets = targets[unlabel_length:]

    kinetics_loss = F.cross_entropy(unlabel_logits, unlabel_targets)
    ucf_loss = F.cross_entropy(label_logits, label_targets)

    return kinetics_loss, ucf_loss, unlabel_length


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class Logger(object):
    """Logger object for recording training process to log file,
    and it supports resume training"""
    def __init__(self, path, header, resume=False):
        """Constructor of this logger

        :param path: logging file path
        :param header: a list of tags for values to track
        :param resume: boolean flag that controls whether to create a new file
                    or continue recording after the latest step
        """
        self.log_file = None
        self.resume = resume
        self.header = header
        if not self.resume:
            self.log_file = open(path, 'w')
            self.logger = csv.writer(self.log_file, delimiter='\t')
            self.logger.writerow(self.header)
        else:
            self.log_file = open(path, 'a+')
            self.log_file.seek(0, os.SEEK_SET)
            reader = csv.reader(self.log_file, delimiter='\t')
            self.header = next(reader)
            # move back to the end of file
            self.log_file.seek(0, os.SEEK_END)
            self.logger = csv.writer(self.log_file, delimiter='\t')

    def __del(self):
        self.log_file.close()

    def log(self, values):
        """log method

        :param values: a dict of values, the keys() of values needs to
                    match the header
        """
        write_values = []
        for tag in self.header:
            assert tag in values, 'Please give the right value as defined!'
            write_values.append(values[tag])

        self.logger.writerow(write_values)
        self.log_file.flush()


def save_checkpoint(state, is_best, opt):
    torch.save(state, os.path.join(opt.result_path, opt.store_name, 'combined_checkpoint.pth'))
    if is_best:
        shutil.copyfile(os.path.join(opt.result_path, opt.store_name, 'combined_checkpoint.pth'),
                        os.path.join(opt.result_path, opt.store_name, 'combined_best_checkpoint.pth'))


def save_ensemble(state, opt):
    torch.save(state, os.path.join(opt.result_path, opt.store_name, 'ensemble_{}.pth'.format(state['past_epochs'])))


def adjust_learning_rate(optimizer, epoch, opt):
    """Set the learning rate to the initial LR decayed by 10 every lr_steps"""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(opt.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new


class NormalizeLayer(nn.Module):
    def __init__(self, power=2):
        super(NormalizeLayer, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
