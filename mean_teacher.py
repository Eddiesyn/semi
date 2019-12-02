import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import time
import json
import numpy as np
import torch.nn.functional as F
from datetime import datetime

from mean import get_mean, get_std
from models import resnet
from dataset import UCF_with_Kinetics
from dataset import get_validation_set
from spatial_transforms import Normalize, MultiScaleRandomCrop, RandomHorizontalFlip
from spatial_transforms import Compose, ToTensor, Scale, CenterCrop
from temporal_transforms import TemporalRandomCrop, TemporalCenterCrop
from target_transforms import ClassLabel, ClassLabel_fromarray
from utils import AverageMeter, calculate_accuracy, Logger, FourStreamSampler


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_path', default='', type=str, help='result path for saving result')
    parser.add_argument('--resume_path', default=None, type=str, help='pth for resume training')
    parser.add_argument('--pretrain_path', default=None, type=str, help='path of pretrained model using UCF101')
    parser.add_argument('--pesudo_label_file', default='', type=str,
                        help='path of pesudo label file')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--init_lr', default=0.05, type=float, help='initial lr')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--dampening', default=0.9, type=float)
    parser.add_argument('--lr_patience', default=1, type=int)
    parser.add_argument('--n_epochs', default=50, type=int)
    # dataset config
    parser.add_argument('--l_vids_path', default='', type=str, help='ucf vids path')
    parser.add_argument('--l_annotation_path', default='', type=str)
    parser.add_argument('--ul_vids_path', default='', type=str)
    parser.add_argument('--ul_annotation_path', default='', type=str)
    parser.add_argument('--norm_value', default=1, type=int)
    parser.add_argument('--mean_dataset', default='activitynet', type=str)
    parser.add_argument('--num_classes', default=101, type=int,
                        help='ucf classes')
    parser.add_argument('--sample_size', default=112, type=int, help='spatial size of input')
    parser.add_argument('--sample_duration', default=16, type=int, help='temporal size of input')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='notice the real batch size is batch size / 2 * 3, so default is 64')
    parser.add_argument('--n_threads', default=8, type=int, help='multi threads for loading data')
    parser.add_argument('--downsample', default=1, type=int, help='Downsampling. Selecting 1 frame out of N')
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--train_crop', default='corner', type=str,
                        help='Spatial cropping method in training. random is uniform. \
                        corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--n_val_samples', default=3, type=int, help='Number of validation samples for each activity')

    parser.add_argument('--manual_seed', default=1, type=int, help='torch random seed')

    args = parser.parse_args()

    return args


def create_model(args, ema=False):
    model = resnet.resnet18(num_classes=args.num_classes,
                            shortcut_type='A',
                            sample_size=args.sample_size,
                            sample_duration=args.sample_duration)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    if args.pretrain_path is not None:
        pretrain = torch.load(args.pretrain_path)
        model.load_state_dict(pretrain['state_dict'])
        old_best = pretrain['best_prec1']
        print('\nThe past best prec1 is {:.5f}'.format(old_best))
    if ema:
        for param in model.parameters():
            param.detach_()

    return model


def feature_mse_loss(f1, f2):
    assert f1.size() == f2.size()
    # Here assume f1, f2 is already l2-normed

    return F.mse_loss(f1, f2, reduction='sum') / f1.size(0)


class CosineLoss(nn.Module):
    def __init__(self, dim=1, eps=1e-08):
        super(CosineLoss, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, f1, f2):
        loss = -F.cosine_similarity(f1, f2, dim=self.dim, eps=self.eps) + 1

        return loss.mean()


# this is simply the nn.MSELoss, use that instead
class FeatureMSE(nn.Module):
    def __init__(self):
        super(FeatureMSE, self).__init__()

    def forward(self, f1, f2):
        loss = F.mse_loss(f1, f2, reduction='mean')


def train(epoch, loader, student_model, teacher_model, optimizer, criterion, batch_logger, epoch_logger, writer):
    print('\nTrain at epoch: {}'.format(epoch + 1))

    student_model.train()
    teacher_model.train()  # ema_model still need at train mode although no grads are computed

    batch_time = AverageMeter()
    data_time = AverageMeter()
    CE_meter = AverageMeter()  # cross_entropy loss meter
    CS_meter = AverageMeter()  # consistency loss meter
    ema_CE_meter = AverageMeter()  # ema_model's cross_entropy loss meter
#    ema_CS_meter = AverageMeter()  # ema_model's consistency loss meter
    top1 = AverageMeter()
    top5 = AverageMeter()
    ema_top1 = AverageMeter()
    ema_top5 = AverageMeter()
    lr_meter = AverageMeter()  # learning rate meter

    end_time = time.time()
    for step, (inputs, targets) in enumerate(loader):
        data_time.update(time.time() - end_time)
        lr_meter.update(optimizer.param_groups[0]['lr'])
        """
        Now inputs have 4 parts each of which has args.batch_size:
        part1&2: kinetics samples with different augmentation
        part3&4: ucf samples with different augmentation
        """
        # this batch_size is the true batch size that is 3/2 of that is specified in terminal
        batch_size = inputs['clip'].size(0) # this should be 4*args.batch_size
        assert batch_size % 4 == 0, 'Invalid batch'
        tiny_batch_size = batch_size // 4

        kinetics_clips = inputs['clip'][:2*tiny_batch_size, :].cuda() # 64+64 kinetics clips
        ucf_clips = inputs['clip'][2*tiny_batch_size:, :].cuda() # 64+64 ucf clips
        ucf_targets = targets['label'][2*tiny_batch_size:].cuda() # 64+64 ucf targets
        # print('kinetics clips: {}, ucf_clips: {}, ucf_targets: {}'.format(kinetics_clips.shape, ucf_clips.shape, ucf_targets.shape))
        assert torch.all(torch.eq(ucf_targets[:tiny_batch_size], ucf_targets[tiny_batch_size:])), 'ucf targets are inconsistent between student and teacher'
        # kinetics_targets is irrelevant
        """
        Construct two inputs: each has half ucf_clips and kinetics_clips and with batchsize
        For student model (model needs grads), ucf_clips is for ce_loss and
        kinetics_clips for cs_loss.
        For teacher model (model without grads), ucf_clips is for evaluation and
        kinetics_clips for cs_loss.
        """
        student_input = torch.cat([ucf_clips[:tiny_batch_size, :], kinetics_clips[:tiny_batch_size, :]], dim=0)
        teacher_input = torch.cat([ucf_clips[tiny_batch_size:], kinetics_clips[tiny_batch_size:, :]], dim=0)

        student_output, student_features = student_model(student_input)
        teacher_output, teacher_features = teacher_model(teacher_input)

        # import pdb; pdb.set_trace()

        # student model's ce_loss and cs_loss
        ce_loss = criterion['classification'](student_output[:tiny_batch_size, :], ucf_targets[:tiny_batch_size]) # first 64 ucf clips to ce_loss
        # first part: kinetics consistency loss, second part: ucf consistency loss
        cs_loss = criterion['consistency'](student_features[tiny_batch_size:, :], teacher_features[tiny_batch_size:, :]) + \
            criterion['consistency'](student_features[:tiny_batch_size, :], teacher_features[:tiny_batch_size, :])

        # teacher model's ce_loss and cs_loss
        # ema_ce_loss here only concerns the ucf clips
        ema_ce_loss = criterion['classification'](teacher_output[:tiny_batch_size, :], ucf_targets[tiny_batch_size:])
        # here ucf_targets[:tiny_batch_size] == ucf_targets[tiny_batch_size:] use any one is ok
        assert not (np.isnan(ce_loss.item()) or ce_loss.item() > 1e5), 'Loss diverge {}!'.format(ce_loss.item())
        assert not (np.isnan(cs_loss.item()) or cs_loss.item() > 1e5), 'Loss diverge {}!'.format(cs_loss.item())

        loss = ce_loss + cs_loss
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss diverge {}!'.format(loss.item())
        print('CE Loss: {}, CS Loss: {}'.format(ce_loss.item(), cs_loss.item())) # only for checking at the beginning, deprecate after ok

        CE_meter.update(ce_loss.item(), tiny_batch_size)
        CS_meter.update(cs_loss.item(), tiny_batch_size)
        ema_CE_meter.update(ema_ce_loss.item(), tiny_batch_size)

        prec1, prec5 = calculate_accuracy(student_output[:tiny_batch_size, :].detach(), ucf_targets[:tiny_batch_size].detach(), topk=(1, 5))
        ema_prec1, ema_prec5 = calculate_accuracy(teacher_output[:tiny_batch_size, :].detach(), ucf_targets[tiny_batch_size:].detach(), topk=(1, 5))
        top1.update(prec1.item(), tiny_batch_size)
        top5.update(prec5.item(), tiny_batch_size)
        ema_top1.update(ema_prec1.item(), tiny_batch_size)
        ema_top5.update(ema_prec5.item(), tiny_batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step = epoch * len(loader) + step + 1
        update_ema_variable(student_model, teacher_model, 0.99, global_step)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # record val of each step
        batch_logger.log({
            'epoch': epoch+1,
            'batch': step+1,
            'iter': global_step,
            'ce_loss': CE_meter.val,
            'cs_loss': CS_meter.val,
            'ema_ce_loss': ema_CE_meter.val,
            'prec1': top1.val,
            'prec5': top5.val,
            'ema_prec1': ema_top1.val,
            'ema_prec5': ema_top5.val,
            'lr': lr_meter.val
        })

        # stdout and tensorboard the avg value every 100 steps
        if (step+1) % 100 == 0:
            print('Epoch [{0}][{1}]/[{2}]  lr {lr:.5f}  '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
                  'ce_loss {CE_meter.val:.4f} ({CE_meter.avg:.4f})  '
                  'cs_loss {CS_meter.val:.4f} ({CS_meter.avg:.4f})  '
                  'ema_ce_loss{ema_CE_meter.val:.4f} ({ema_CE_meter.avg:.4f})  '
                  'prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
                  'prec@5 {top5.val:.3f} ({top5.avg:.3f})  '
                  'ema_prec@1 {ema_top1.val:.3f} ({ema_top1.avg:.3f})  '
                  'ema_prec@5 {ema_top5.val:.3f} ({ema_top5.avg:.3f})'.format(
                      epoch+1,
                      step+1,
                      len(loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      CE_meter=CE_meter,
                      CS_meter=CS_meter,
                      ema_CE_meter=ema_CE_meter,
                      top1=top1,
                      top5=top5,
                      ema_top1=ema_top1,
                      ema_top5=ema_top5,
                      lr=lr_meter.val
                  ))
            writer.add_scalar('Train/Student/ce_loss', CE_meter.avg, global_step)
            writer.add_scalar('Train/Student/cs_loss', CS_meter.avg, global_step)
            writer.add_scalar('Train/Student/prec1', top1.avg, global_step)
            writer.add_scalar('Train/Student/prec5', top5.avg, global_step)

            writer.add_scalar('Train/Teacher/ce_loss', ema_CE_meter.avg, global_step)
            writer.add_scalar('Train/Teacher/prec1', ema_top1.avg, global_step)
            writer.add_scalar('Train/Teacher/prec5', ema_top5.avg, global_step)

    epoch_logger.log({
        'epoch': epoch+1,
        'ce_loss': CE_meter.avg,
        'cs_loss': CS_meter.avg,
        'ema_ce_loss': ema_CE_meter.avg,
        'prec1': top1.avg,
        'prec5': top5.avg,
        'ema_prec1': ema_top1.avg,
        'ema_prec5': ema_top5.avg,
        'lr': lr_meter.avg
    })


def validate(epoch, student_model, teacher_model, loader, criterion, val_logger, writer):
    print('\nValidate after epoch {}'.format(epoch + 1))

    student_model.eval()
    teacher_model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    CE_meter = AverageMeter()
    ema_CE_meter = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    ema_top1 = AverageMeter()
    ema_top5 = AverageMeter()

    end_time = time.time()
    for step, (inputs, targets) in enumerate(loader):
        data_time.update(time.time() - end_time)

        """
        validation set only yields ucf_clips
        """
        with torch.no_grad():
            inputs = inputs.cuda()
            targets = targets.cuda()
        student_outputs, _ = student_model(inputs)
        teacher_outputs, _ = teacher_model(inputs)

        ce_loss = criterion['classification'](student_outputs, targets)
        ema_ce_loss = criterion['classification'](teacher_outputs, targets)
        CE_meter.update(ce_loss.item(), inputs.size(0))
        ema_CE_meter.update(ema_ce_loss.item(), inputs.size(0))

        prec1, prec5 = calculate_accuracy(student_outputs.detach(), targets.detach(), topk=(1, 5))
        ema_prec1, ema_prec5 = calculate_accuracy(teacher_outputs.detach(), targets.detach(), topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        ema_top1.update(ema_prec1.item(), inputs.size(0))
        ema_top5.update(ema_prec5.item(), inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]  '
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
              'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
              'ce_loss {CE_meter.val:4f} ({CE_meter.avg:.4f})  '
              'ema_ce_loss {ema_CE_meter.val:.4f} ({ema_CE_meter.avg:.4f})  '
              'Prec@1 {top1.val:.4f} ({top1.avg:.4f})  '
              'Prec@5 {top5.val:.4f} ({top5.avg:.4f})  '
              'ema_Prec@1 {ema_top1.val:.4f} ({ema_top1.avg:.4f})  '
              'ema_Prec@5 {ema_top5.val:.4f} ({ema_top5.avg:.4f})'.format(
                  epoch+1,
                  step+1,
                  len(loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  CE_meter=CE_meter,
                  ema_CE_meter=ema_CE_meter,
                  top1=top1,
                  top5=top5,
                  ema_top1=ema_top1,
                  ema_top5=ema_top5
              ))
    val_logger.log({
        'epoch': epoch+1,
        'ce_loss': CE_meter.avg,
        'ema_ce_loss': ema_CE_meter.avg,
        'prec1': top1.avg,
        'prec5': top5.avg,
        'ema_prec1': ema_top1.avg,
        'ema_prec5': ema_top5.avg,
    })
    writer.add_scalar('Eval/Student/ce_loss', CE_meter.avg, epoch+1)
    writer.add_scalar('Eval/Student/prec1', top1.avg, epoch+1)
    writer.add_scalar('Eval/Student/prec5', top5.avg, epoch+1)
    writer.add_scalar('Eval/Teacher/ce_loss', ema_CE_meter.avg, epoch+1)
    writer.add_scalar('Eval/Teacher/prec1', ema_top1.avg, epoch+1)
    writer.add_scalar('Eval/Teacher/prec5', ema_top5.avg, epoch+1)

    return CE_meter.avg, ema_CE_meter.avg


def update_ema_variable(student_model, teacher_model, alpha, global_step):
    # Use the true average util the exponential average is more correct
    alpha = min(1 - (1 / (global_step + 1)), alpha)
    for ema_param, param in zip(teacher_model.parameters(), student_model.parameters()):
        ema_param.detach().mul_(alpha).add_((1 - alpha) * param.detach())


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if not os.path.exists(os.path.join(args.result_path, 'tensorboard_logs')):
        os.makedirs(os.path.join(args.result_path, 'tensorboard_logs'))
    log_dir = os.path.join(args.result_path, 'tensorboard_logs')
    args.scales = [args.initial_scale]
    # scales for multi-scale cropping
    for i in range(1, args.n_scales):
        args.scales.append(args.scales[-1] * args.scale_step)
    # mean, std of dataset
    args.mean = get_mean(args.norm_value, dataset=args.mean_dataset)
    args.std = get_std(args.norm_value)

    print(args)
    with open(os.path.join(args.result_path, 'args.json'), 'w') as args_file:
        json.dump(vars(args), args_file)

    torch.manual_seed(args.manual_seed)
    labels = np.load(args.pesudo_label_file)

    # get teacher and student model
    student_model = create_model(args)
    teacher_model = create_model(args, ema=True)

    """
    Prepare dataset and loader for mean-teacher training
    Here each batch contains three parts, each has same number of samples.
    Parts 1: ucf101 samples, contribute to cross_entropy loss
    Parts 2&3: kinetics samples with different augmentation, contribute to consistency loss

    right now only consider augmentation on temporal dimension,
    since i think temporal information contributes more to action recognition than spatial information
    """
    norm_method = Normalize(args.mean, [1, 1, 1])
    ucf_crop = MultiScaleRandomCrop(args.scales, args.sample_size)
    ucf_spatial = Compose([
        RandomHorizontalFlip(),
        ucf_crop,
        ToTensor(args.norm_value), norm_method,
    ])
    kinetics_spatial = Compose([
        RandomHorizontalFlip(),
        ucf_crop,
        ToTensor(args.norm_value), norm_method,
    ])

    ucf_temporal = TemporalRandomCrop(args.sample_duration, args.downsample)
    # ucf_temporal = TemporalCenterCrop(args.sample_duration, args.downsample)
#    kinetics_temporal = TransformTwice(TemporalRandomCrop(args.sample_duration, args.downsample))
    kinetics_temporal = TemporalRandomCrop(args.sample_duration, args.downsample)
#     kinetics_temporal = TemporalCenterCrop(args.sample_duration, args.downsample)

    spatial_transform = [ucf_spatial, kinetics_spatial]
    temporal_transform = [ucf_temporal, kinetics_temporal]
    target_transform = [ClassLabel(), ClassLabel_fromarray(labels)]
    # here the second is irrelevant since we don't use their labels

    combined_dataset = UCF_with_Kinetics(args.l_vids_path,
                                         args.l_annotation_path,
                                         args.ul_vids_path,
                                         args.ul_annotation_path,
                                         'training',
                                         1,
                                         spatial_transform=spatial_transform,
                                         temporal_transform=temporal_transform,
                                         target_transform=target_transform,
                                         sample_duration=args.sample_duration)
    label_length = combined_dataset.labeled_length
    unlabel_length = combined_dataset.unlabeled_length
    assert label_length + unlabel_length == len(combined_dataset), 'Fatal Error!'
    print('\nTotal: {} labeled samples and {} unlabeled samples'.format(label_length, unlabel_length))

    val_sp_transform = Compose([
        Scale(args.sample_size),
        CenterCrop(args.sample_size),
        ToTensor(args.norm_value), norm_method,
    ])
    val_tp_transform = TemporalCenterCrop(args.sample_duration, args.downsample)
    val_dataset = get_validation_set(args, val_sp_transform, val_tp_transform, target_transform[0])

    # generate loader
    label_indices = torch.from_numpy(np.arange(label_length))
    unlabel_indices = torch.from_numpy(np.arange(label_length, len(combined_dataset)))
    """A batchsampler yielding a batch of indices with 4 parts, 
    1&2 from primary indices(kinetics), 
    3&4 from secondary indices(ucf), 
    1 and 2 are same indices (they are same samples), but each is augmented differently (spatially and temporally)
    3 and 4 same as above
    batch_size should be 64 so the 1&3 will be fed to student model (with batchsize 128)
    2&4 will be fed to teacher model (with batchsize 128)
    """
    sampler = FourStreamSampler(unlabel_indices, label_indices, args.batch_size, args.batch_size)
    train_loader = torch.utils.data.DataLoader(dataset=combined_dataset,
                                               batch_sampler=sampler,
                                               num_workers=args.n_threads,
                                               pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=16,
                                              shuffle=False,
                                              num_workers=args.n_threads,
                                              pin_memory=True)

    # # do loader test, comment when succeeds
    # for step, (inputs, targets) in enumerate(train_loader):
    #     if step == 3:
    #        break
    #     """Now inputs should have 3 parts each has batch_size/2
    #     part1&2: kinetics samples with different augmentation
    #     part3&4: ucf samples with different augmentation
    #     """
    #     batch_size = inputs['clip'].size(0)
    #     print('A batch has {} samples'.format(batch_size))
    #     # assert batch_size % 3 == 0, 'Invalid batch'
    #     # kinetics_size = int(batch_size/3*2)
    #     assert batch_size % 4 == 0, 'Invalid batch'
    #     kinetics_size = batch_size // 2
    #     tiny_batch_size = batch_size // 4 # this should equal to args.batch_size
    #     print('step {}'.format(step))
    #     # check if they are different clips
    #     kinetics_indices = inputs['frame_indices'][:kinetics_size, :]  # (128, 16)
    #     ucf_indices = inputs['frame_indices'][kinetics_size:, :]  # (128, 16)
    #     print('two kinetics indices')
    #     print(kinetics_indices[:args.batch_size, :])
    #     print(kinetics_indices[args.batch_size:, :])
    #     print('two ucf indices')
    #     print(ucf_indices[:args.batch_size, :])
    #     print(ucf_indices[args.batch_size:, :])
    #     # check if they come from same samples
    #     kinetics_targets = targets['label'][:kinetics_size]
    #     ucf_targets = targets['label'][kinetics_size:]
    #     print('two kinetics targets')
    #     print(kinetics_targets[:args.batch_size])
    #     print(kinetics_targets[args.batch_size:])
    #     print('two ucf targets')
    #     print(ucf_targets[:args.batch_size])
    #     print(ucf_targets[args.batch_size:])
    #
    #     # save and later on check if they are spatially differently augmented
    #     kinetics_clips = inputs['clip'][:kinetics_size, :]
    #     ucf_clips = inputs['clip'][kinetics_size:, :]
    #
    #     np.save(os.path.join(args.result_path, 'kinetics_clips_{}'.format(step)), kinetics_clips.numpy())
    #     np.save(os.path.join(args.result_path, 'ucf_clips_{}'.format(step)), ucf_clips.numpy())
    #
    #     print('==============')

    # prepare criterion and optimizer
    """
    The consistency_loss is calculated using logits while here use
    features, since the consistency_loss use samples from a different
    source other than the samples which calculates cross_entropy loss.
    Since no softmax is applied before calculating consistency_loss, kl
    divergence is unsuitable for measuring distance in feature space.
    We use mse as default which is basically the euclidean distance in
    feature space. According to researches on face verification, a L2-norm
    on CNN features could help since it can optimize it to cosine similarity
    which is more suitable for measuring similarity in high-dimensional
    feature space.
    """
    class_criterion = nn.CrossEntropyLoss().cuda()
    # consistency_criterion = CosineLoss().cuda()
    consistency_criterion = nn.MSELoss().cuda()
    criterion = {}
    criterion['classification'] = class_criterion
    criterion['consistency'] = consistency_criterion

    if args.nesterov:
        dampening = 0
    else:
        dampening = args.dampening

    optimizer = torch.optim.SGD(student_model.parameters(),
                                args.init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    # scheduler focuses on val loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           patience=args.lr_patience,
                                                           verbose=True,
                                                           threshold=0.01,
                                                           threshold_mode='abs')
    now = datetime.now()
    # tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(log_dir, now.strftime("%Y%m%d-%H%M%S")))

    begin_epoch = 0
    resume = False
    if args.resume_path is not None:
        resume = True
        print('\nResume training...')
        resume_dict = torch.load(args.resume_path)
#        import pdb; pdb.set_trace()
        student_model.load_state_dict(resume_dict['student_state_dict'])
        teacher_model.load_state_dict(resume_dict['teacher_state_dict'])
#        student_model = DecoupleModel(student_model)
#        teacher_model = DecoupleModel(teacher_model)
        begin_epoch = resume_dict['epoch']
        optimizer.load_state_dict(resume_dict['optimizer'])

    batch_logger_names = ['epoch', 'batch', 'iter', 'ce_loss', 'cs_loss', 'ema_ce_loss', 'prec1', 'prec5', 'ema_prec1', 'ema_prec5', 'lr']
    batch_logger = Logger(os.path.join(args.result_path, 'train_batch.log'), batch_logger_names, resume=resume)
    epoch_logger_names = ['epoch', 'ce_loss', 'cs_loss', 'ema_ce_loss', 'prec1', 'prec5', 'ema_prec1', 'ema_prec5', 'lr']
    epoch_logger = Logger(os.path.join(args.result_path, 'train.log'), epoch_logger_names, resume=resume)
    val_logger = Logger(os.path.join(args.result_path, 'val.log'),
                        ['epoch', 'ce_loss', 'ema_ce_loss', 'prec1', 'prec5', 'ema_prec1', 'ema_prec5'])

    for i in range(begin_epoch, args.n_epochs):
        train(i, train_loader, student_model, teacher_model, optimizer, criterion, batch_logger, epoch_logger, writer)
        # save every 1 epoch
        states = {
            'epoch': i+1,
            'student_state_dict': student_model.state_dict(),
            'teacher_state_dict': teacher_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, os.path.join(args.result_path, 'epoch_{}.pth'.format(i+1)))
        # do evaluation and monitor val ce loss, here use student and teacher both to check
        # the performace of the EMA
        student_ce_loss, teacher_ce_loss = validate(i, student_model, teacher_model, eval_loader, criterion, val_logger, writer)
        scheduler.step(student_ce_loss)
