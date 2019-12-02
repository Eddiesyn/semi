import os
import json
from collections import OrderedDict
from itertools import islice
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from opts import parse_opts
from model import generate_combined_model, generate_model, get_model
from model import get_original_model
from spatial_transforms import Normalize, MultiScaleCornerCrop, MultiScaleRandomCrop
from spatial_transforms import Compose, ToTensor, RandomHorizontalFlip
from spatial_transforms import Scale, CenterCrop
from temporal_transforms import TemporalCenterCrop, TemporalRandomCrop
from target_transforms import ClassLabel, ClassLabel_fromarray
from mean import get_mean, get_std
from dataset import prepare_ucf_and_kinetics, generate_combined_loader, get_validation_set
from dataset import Kinetics_clustered
from utils import Logger, save_checkpoint, save_ensemble
from train import fine_tune_on_both, val_epoch, train_on_clusters


if __name__ == '__main__':
    opt = parse_opts()
#    if not os.path.exists(os.path.join(opt.result_path, opt.store_name)):
#        os.makedirs(os.path.join(opt.result_path, opt.store_name))

    # multi scale cropping
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, opt.store_name, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    # read the generated pesudo label file for fine-tuning
    labels = np.load(opt.pesudo_label_file)
    num_new_classes = len(np.unique(labels))
    print('Use {} clusters for unlabeled datasets supervision'.format(num_new_classes))

    # build model (the original pretrained one) and decouple the top layer from it
    model = get_model(opt)
    old_fc = model.module.fc
    model.module.fc = nn.Linear(model.module.fc.in_features, num_new_classes)
    model.module.fc = model.module.fc.cuda()

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    optimizer = optim.SGD(
        model.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov
    )

#    optimizer = optim.SGD([
#        {'params': model.old_model.parameters(), 'lr': 1e-3},
#        {'params': model.new_fc.parameters(), 'lr': 0.1}],
#        momentum=opt.momentum,
#        dampening=dampening,
#        weight_decay=opt.weight_decay,
#        nesterov=opt.nesterov
#    )
#    import pdb; pdb.set_trace()

#    optimizer = optim.Adam(model.parameters(),
#                           lr=opt.learning_rate,
#                           weight_decay=opt.weight_decay)

#    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                     mode='min',
#                                                     patience=opt.lr_patience,
#                                                     verbose=True,
#                                                     threshold=0.01,
#                                                     threshold_mode='abs')
    # do lr decay every 1 epoch (rate is sqrt(0.1))
    decay_rate = 0.316
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=decay_rate)

    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    assert opt.train_crop in ['random', 'corner', 'center']
    if opt.train_crop == 'random':
        ucf_crop = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'corner':
        ucf_crop = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'center':
        ucf_crop = MultiScaleCornerCrop(opt.scales, opt.sample_size, crop_positions=['c'])

    spatial_transform = []
    temporal_transform = []
    target_transform = []

    ucf_transform = Compose([
        RandomHorizontalFlip(),
        ucf_crop,
        ToTensor(opt.norm_value), norm_method,
    ])

    kinetics_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        ToTensor(opt.norm_value), norm_method,
    ])

    spatial_transform.append(ucf_transform)
    spatial_transform.append(kinetics_transform)

    temporal_transform.append(TemporalRandomCrop(opt.sample_duration, opt.downsample))
    temporal_transform.append(TemporalCenterCrop(opt.sample_duration, opt.downsample))

    target_transform.append(ClassLabel())
    target_transform.append(ClassLabel_fromarray(labels))

    kinetics_clustered = Kinetics_clustered(opt.ul_vids_path,
                                            opt.ul_annotation_path,
                                            'training',
                                            1,
                                            spatial_transform=spatial_transform[0],
                                            temporal_transform=temporal_transform[0],
                                            target_transform=target_transform[1])
    train_loader = torch.utils.data.DataLoader(dataset=kinetics_clustered,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.n_threads,
                                               pin_memory=True)

#    combined_dataset, label_length, unlabel_length = prepare_ucf_and_kinetics(
#        opt, spatial_transform, temporal_transform, target_transform)
#    labeled_class_names = combined_dataset.labeled_class_names
#
#    val_sp_transform = Compose([
#        Scale(opt.sample_size),
#        CenterCrop(opt.sample_size),
#        ToTensor(opt.norm_value), norm_method
#    ])
#    val_dataset = get_validation_set(opt, val_sp_transform, temporal_transform[1], target_transform[0])
#    train_loader, eval_loader = generate_combined_loader(opt, combined_dataset, val_dataset, label_length)

    now = datetime.now()
    writer = SummaryWriter(log_dir=os.path.join(opt.result_path, opt.store_name,
                                                'tensorboard_logs', now.strftime("%Y%m%d-%H%M%S")))

#    train_logger = Logger(os.path.join(opt.result_path, opt.store_name, 'train.log'),
#                          ['epoch', 'kinetics_loss', 'ucf_loss', 'prec1', 'prec5', 'old_lr', 'new_lr'])
#    train_batch_logger = Logger(os.path.join(opt.result_path, opt.store_name, 'train_batch.log'),
#                                ['epoch', 'batch', 'iter', 'kinetics_loss', 'ucf_loss', 'prec1', 'prec5', 'old_lr', 'new_lr'])
#    val_logger = Logger(os.path.join(opt.result_path, opt.store_name, 'val.log'),
#                        ['epoch', 'kinetics_loss', 'ucf_loss', 'prec1', 'prec5'])
#    train_batch_logger = Logger(os.path.join(opt.result_path, opt.store_name, 'train_batch.log'),
#                                ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])
#    train_logger = Logger(os.path.join(opt.result_path, opt.store_name, 'train.log'),
#                          ['epoch', 'loss', 'prec1', 'lr'])
#
#    best_prec1 = old_best
#    train_on_clusters(opt, 0, train_loader, model, criterion, optimizer, train_logger, train_batch_logger, writer)

    # first train 5 epochs till loss is saturating
    begin_epoch = 0
    resume = False
    if opt.resume_path is not None:
        resume = True
        print('\nResume training....')
        resume = torch.load(opt.resume_path)
        model.load_state_dict(resume['state_dict'])
        begin_epoch = resume['epoch']
        optimizer.load_state_dict(resume['optimizer'])

        # temp code snippet, will be deprecate
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.learning_rate
#        scheduler.step(1.8985)

    batch_header = ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr']
    epoch_header = ['epoch', 'loss', 'prec1', 'lr']

    train_batch_logger = Logger(os.path.join(opt.result_path, opt.store_name, 'train_batch.log'), batch_header, resume)
    train_epoch_logger = Logger(os.path.join(opt.result_path, opt.store_name, 'train.log'), epoch_header, resume)

    for i in range(begin_epoch, opt.n_epochs):
        past_epochs = i+1
        epoch_loss, epoch_prec1, epoch_prec5 = train_on_clusters(opt,
                                                                 i,
                                                                 train_loader,
                                                                 model,
                                                                 criterion,
                                                                 optimizer,
                                                                 train_epoch_logger,
                                                                 train_batch_logger,
                                                                 writer)
        # scheduler.step(epoch_loss)
        scheduler.step()

        new_fc = model.module.fc
        model_state_dict = model.state_dict()
        backbone_state_dict = islice(model_state_dict.items(), len(model_state_dict)-2)
        backbone_state_dict = OrderedDict(backbone_state_dict)
        state = {
            'past_epochs': past_epochs,
            'backbone': backbone_state_dict,
            'new_fc': new_fc.state_dict(),
            'old_fc': old_fc.state_dict(),
        }
        save_ensemble(state, opt)

    # then save for next cluster
#    fine_tune_on_both(opt, 0, train_loader, model, criterion, optimizer, train_logger, train_batch_logger, writer)
#    for i in range(opt.n_epochs):
#        fine_tune_on_both(opt, i, train_loader, model, criterion, optimizer, train_logger, train_batch_logger, writer)
#        state = {
#            'epoch': i+1,
#            'state_dict': model.state_dict(),
#            'optimizer': optimizer.state_dict(),
#            'best_prec1': best_prec1
#        }
#        save_checkpoint(state, False, opt)
#
#        prec1 = val_epoch(i, eval_loader, model, criterion, opt, val_logger, writer)
#        is_best = prec1 > best_prec1
#        state = {
#            'epoch': i+1,
#            'state_dict': model.state_dict(),
#            'optimizer': optimizer.state_dict(),
#            'best_prec1': best_prec1
#        }
#        save_checkpoint(state, is_best, opt)
