from torch.utils.data import Dataset
import torch
import numpy as np

from datasets import ucf101, kinetics
from utils import TwoStreamSampler


class UCF_with_Kinetics(Dataset):
    """Dataset class for combining kinetics and ucf into one batch

    Args:
        labeled_root_path (str): directory of labeled dataset
        labeled_annotation_path (str): annotation file of labeled dataset
        unlabeled_root_path (str): directory of unlabeled dataset
        unlabeled_annotation_path (str): annotation file of unlabeled dataset
        subset (str): either 'training' or 'validation'
        n_samples_for_each_video (int): extracted number of samples of each vids
        spatial_transform (list): a list of two transformation, the first
            belongs to labeled dataset and the second belongs to unlabeled dataset
        temporal_transform (list): same as above for temporal transformation
        target_transform (list): same as above for target transformation
        sample_duration (int): temporal length of each sample
        labeled_vids_loader (function):
        unlabeled_vids_loader (function):

    """
    def __init__(self,
                 labeled_root_path,
                 labeled_annotation_path,
                 unlabeled_root_path,
                 unlabeled_annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 labeled_vids_loader=ucf101.get_default_video_loader,
                 unlabeled_vids_loader=kinetics.get_default_video_loader):
        self.data, self.labeled_class_names = ucf101.make_dataset(
            labeled_root_path, labeled_annotation_path, subset, n_samples_for_each_video, sample_duration
        )
        self.unlabeled_data, _ = kinetics.make_dataset(unlabeled_root_path,
                                                       unlabeled_annotation_path,
                                                       subset, n_samples_for_each_video, sample_duration)

        self.labeled_length = len(self.data)  # for secondary iter (labeled)
        self.unlabeled_length = len(self.unlabeled_data)  # for primary iter (unlabeled)

        # concatenate two data list, [ucf_1, ..., kinetics_1, ...]
        self.data.extend(self.unlabeled_data)

        self.label_loader = labeled_vids_loader()
        self.unlabeled_loader = unlabeled_vids_loader()

        self.label_transform = {
            'spatial_transform': spatial_transform[0],
            'temporal_transform': temporal_transform[0],
            'target_transform': target_transform[0],
        }
        self.unlabel_transform = {
            'spatial_transform': spatial_transform[1],
            'temporal_transform': temporal_transform[1],
            'target_transform': target_transform[1],
        }

        self.sample_duration = sample_duration

    def __getitem__(self, index):
        path = self.data[index]['video']
        ensemble_target = {}
        ensemble_clip = {}

        if index < self.labeled_length:
            # is ucf vids data
            frame_indices = self.data[index]['frame_indices']
            frame_indices = self.label_transform['temporal_transform'](frame_indices)
            ensemble_clip['frame_indices'] = torch.from_numpy(np.array(frame_indices))

            clip = self.label_loader(path, frame_indices)
            target = self.data[index]

            if self.label_transform['target_transform'] is not None:
                target = self.label_transform['target_transform'](target)
                ensemble_target['label'] = target
                ensemble_target['source'] = 'ucf101'
            if self.label_transform['spatial_transform'] is not None:
                self.label_transform['spatial_transform'].randomize_parameters()
                clip = [self.label_transform['spatial_transform'](img) for img in clip]
                clip = torch.stack(clip, 0).permute(1, 0, 2, 3)  # C D H W
        else:
            # is kinetics vids data
            # index -= self.labeled_length
            frame_indices = self.data[index]['frame_indices']
            frame_indices = self.unlabel_transform['temporal_transform'](frame_indices)

            clip = self.unlabeled_loader(path, frame_indices, self.sample_duration)
            ensemble_clip['frame_indices'] = torch.from_numpy(np.array(frame_indices))

            if self.unlabel_transform['target_transform'] is not None:
                # print(index - self.labeled_length)
                target = self.unlabel_transform['target_transform'](index-self.labeled_length)
                ensemble_target['label'] = target
                ensemble_target['source'] = 'kinetics'
            if self.unlabel_transform['spatial_transform'] is not None:
                self.unlabel_transform['spatial_transform'].randomize_parameters()
                clip = [self.unlabel_transform['spatial_transform'](img) for img in clip]
                clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        ensemble_clip['clip'] = clip

        return ensemble_clip, ensemble_target

#        return clip, ensemble_target
#        return clip, target

    def __len__(self):
        return len(self.data)


# about to deprecate
class Kinetics_clustered(Dataset):
    def __init__(self,
                 unlabeled_root_path,
                 unlabeled_annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 loader=kinetics.get_default_video_loader):
        self.data, _ = kinetics.make_dataset(unlabeled_root_path,
                                             unlabeled_annotation_path,
                                             subset, n_samples_for_each_video, sample_duration)
        self.sample_duration = sample_duration
        self.loader = loader()
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        assert self.temporal_transform is not None, 'Please give a valid temporal transform!'
        self.target_transform = target_transform
        # assert self.target_transform is not None, 'Please give a valid target transform!'

    def __getitem__(self, index):
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices, self.sample_duration)
        target = self.data[index]

        if self.target_transform is not None:
            target = self.target_transform(index)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip, target

    def __len__(self):
        return len(self.data)


def prepare_ucf_and_kinetics(opt, spatial_transform, temporal_transform, target_transform):
    """Prepare the combined ucf and kinetics dataloader

    :param opt: from argparse
    :param spatial_transform: a list of two transformation, first for labeled, second for unlabeled
    :param temporal_transform: same as above for temporal transformation
    :param target_transform: same as above for target transformation

    """
    combined_dataset = UCF_with_Kinetics(opt.l_vids_path,
                                         opt.l_annotation_path,
                                         opt.ul_vids_path,
                                         opt.ul_annotation_path,
                                         'training',
                                         1,
                                         spatial_transform=spatial_transform,
                                         temporal_transform=temporal_transform,
                                         target_transform=target_transform,
                                         sample_duration=16)
    label_length = combined_dataset.labeled_length
    unlabel_length = combined_dataset.unlabeled_length
    assert label_length + unlabel_length == len(combined_dataset), 'Fatal error in initiating dataset'

    return combined_dataset, label_length, unlabel_length


def generate_combined_loader(opt, combined_dataset, val_dataset, label_length):
    label_indices = torch.from_numpy(np.arange(label_length))
    unlabel_indices = torch.from_numpy(np.arange(label_length, len(combined_dataset)))

    '''Add int when passing secondary_batch_size (e.g. labeled batch_size)
    to TwoStreamSampler, the last batch of unlabeled dataset is dropped.

    '''
    sampler = TwoStreamSampler(unlabel_indices, label_indices, opt.batch_size, int(opt.ratio * opt.batch_size))
    train_loader = torch.utils.data.DataLoader(dataset=combined_dataset,
                                               batch_sampler=sampler,
                                               num_workers=opt.n_threads,
                                               pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=16,
                                              shuffle=False,
                                              num_workers=opt.n_threads,
                                              pin_memory=True)

    return train_loader, eval_loader


def get_validation_set(opt, spatial_transform, temporal_transform, target_transform):
    validation_data = ucf101.UCF101(
        opt.l_vids_path,
        opt.l_annotation_path,
        'validation',
        opt.n_val_samples,
        spatial_transform,
        temporal_transform,
        target_transform,
        sample_duration=opt.sample_duration
    )

    return validation_data


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    training_data = ucf101.UCF101(
        opt.l_vids_path,
        opt.l_annotation_path,
        'training',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration
    )

    return training_data
