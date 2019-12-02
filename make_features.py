from sklearn.decomposition import PCA
import time
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import argparse

from utils import Identity, AverageMeter
from mean import get_mean, get_std
from spatial_transforms import ToTensor, Normalize
from temporal_transforms import TemporalCenterCrop
# from target_transforms import ClassLabel_fromarray
from dataset import Kinetics_clustered
from models import resnet


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_components', default=[8, 16, 32, 64, 128], type=int, nargs="+",
                        metavar='num_comp', help='the reduced dimension of features')
    parser.add_argument('--norm_value', default=1, type=int,
                        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1]')
    parser.add_argument('--mean_dataset', default='activitynet', type=str,
                        help='dataset for mean values of mean substraction (activitynet | kinetics)')
    parser.add_argument('--manual_seed', default=1, type=int,
                        help='Manually set random seed')
    parser.add_argument('--pretrain_path', default='', type=str,
                        help='path of pretrained model')
    parser.add_argument('--num_classes', default=101, type=int,
                        help='number of classes of UCF101')
    # args for dataset
    parser.add_argument('--ul_vids_path', default='', type=str,
                        help='video path')
    parser.add_argument('--ul_annotation_path', default='', type=str,
                        help='annotations path')
    parser.add_argument('--sample_size', default=112, type=int,
                        help='Spatial size of inputs')
    parser.add_argument('--sample_duration', default=16, type=int,
                        help='Temporal duration of inputs')
    parser.add_argument('--pesudo_label_file', default='', type=str,
                        help='the clustered label file')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size of data loader')
    parser.add_argument('--n_threads', default=8, type=int,
                        help='number of threads for loading data')
    parser.add_argument('--dst_path', default='', type=str,
                        help='the dst directory of extracted features')
    parser.add_argument('--downsample', default=1, type=int,
                        help='Downsampling. Select number of frames out of N')

    args = parser.parse_args()

    return args


def make_features(dataloader, model, args, N):
    """Extract features and do PCA dim-reduction on the whole
    dataset.

    :param dataloader: loader of the whole kinetics dataset
    :param model: the model has no top layer and output features
    :param args:
    :param N: length of total samples
    """

    batch_time = AverageMeter()
    end = time.time()

    model.eval()  # set model in eval mode

    with torch.no_grad():
        for step, (inputs, _) in enumerate(dataloader):
            inputs = inputs.cuda()

            outputs = model(inputs)  # now this is CNN features
            array = outputs.cpu().numpy()
            # import pdb;pdb.set_trace()

            if step == 0:
                total_features = np.zeros((N, array.shape[1]), dtype='float32')
            if step < len(dataloader) - 1:
                total_features[step * args.batch_size: (step+1) * args.batch_size, :] = array
            else:
                # special treatment for final batch
                total_features[step * args.batch_size:, :] = array

            batch_time.update(time.time() - end)
            end = time.time()

            if step % 100 == 0:
                print('step: [{}]/[{}], batch_time: {:.5f}'.format(
                    step, len(dataloader), batch_time.avg
                ))

    begin_time = time.time()

    np.save(os.path.join(args.dst_path, 'features.npy'), total_features)

    # set pca instance
    for n_comp in args.n_components:
        pca = PCA(n_components=n_comp, svd_solver='auto')
        features = pca.fit_transform(total_features)
        print('PCA to {} dim finished in {:.5f}s'.format(n_comp, time.time() - begin_time))
        np.save(os.path.join(args.dst_path, 'features_{}.npy'.format(n_comp)), features)
        begin_time = time.time()


def get_model(args):
    """Use model pretrained on UCF101
    """
    model = resnet.resnet18(num_classes=args.num_classes,
                            shortcut_type='A',
                            sample_size=args.sample_size,
                            sample_duration=args.sample_duration)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    print('Use pretrained model {}'.format(args.pretrain_path))
    pretrain = torch.load(args.pretrain_path)
    if 'backbone' in pretrain.keys():
        model.module.fc = Identity()
        model.load_state_dict(pretrain['backbone'])
    else:
        model.load_state_dict(pretrain['state_dict'])
        model.module.fc = Identity()

    return model


# def get_ensemble_model(args):
#    """Use ensemble model fine_tuned on clusters
#    """
#    model = resnet.resnet18(num_classes=args.num_classes,
#                            shortcut_type='A',
#                            sample_size=args.sample_size,
#                            sample_duration=args.sample_duration)
#    labels = np.load(args.pesudo_label_file)
#    num_new_classes = len(np.unique(labels))
#    model = model.cuda()
#    model = nn.DataParallel(model, device_ids=None)
#    model.module.fc = nn.Linear(model.module.fc.in_features, num_new_classes)
#    model.module.fc = model.module.fc.cuda()
#
#    print('Use fine_tuned model {}'.format(args.pretrain_path))
#    pretrain = torch.load(args.pretrain_path)
#
#    model.load_state_dict(pretrain['state_dict'])
#
#    return model


if __name__ == '__main__':
    args = get_args()

    args.mean = get_mean(args.norm_value, dataset=args.mean_dataset)
    args.std = get_std(args.norm_value)

    torch.manual_seed(args.manual_seed)

    model = get_model(args)
    # model = get_ensemble_model(args)

    # substract mean and no normalization by default
    norm_method = Normalize(args.mean, [1, 1, 1])

    spatial_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        ToTensor(args.norm_value), norm_method,
    ])

    temporal_transform = TemporalCenterCrop(args.sample_duration, args.downsample)

    # prepare dataset
    kinetics_dataset = Kinetics_clustered(args.ul_vids_path,
                                          args.ul_annotation_path,
                                          'training',
                                          1,
                                          spatial_transform=spatial_transform,
                                          temporal_transform=temporal_transform,
                                          target_transform=None)
    loader = torch.utils.data.DataLoader(dataset=kinetics_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.n_threads,
                                         pin_memory=True)
    N = len(kinetics_dataset)
    print('Try get features from {} samples'.format(N))

    make_features(loader, model, args, N)
#    features = make_features(loader, model, args, N)
#    np.save(os.path.join(args.dst_path, 'features.npy'), features)
