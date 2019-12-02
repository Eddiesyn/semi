from utils import AverageMeter, calculate_accuracy

import time
import os
import torch


def fine_tune_on_both(opt, epoch, data_loader, model, criterion, optimizer, epoch_logger, batch_logger, writer):
    print('fine tune at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    ucf_loss = AverageMeter()
    kinetics_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    for step, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time()-end_time)
        labels = targets['label']
        # sources = targets['source']
        if not opt.no_cuda:
            labels = labels.cuda()
            # targets = targets.cuda()
            inputs = inputs.cuda()

        old_logits, new_logits = model(inputs)
        # new (unlabel) comes first
        new_targets, old_targets = labels[:new_logits.size(0)], labels[new_logits.size(0):]

        label_loss = criterion(old_logits, old_targets)
        unlabel_loss = criterion(new_logits, new_targets)
        total_loss = unlabel_loss + label_loss

        kinetics_loss.update(unlabel_loss.item(), new_logits.size(0))
        ucf_loss.update(label_loss.item(), old_logits.size(0))

        prec1, prec5 = calculate_accuracy(old_logits.detach(), old_targets.detach(), topk=(1, 5))
        top1.update(prec1, old_logits.size(0))
        top5.update(prec5, old_logits.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch+1,  # epoch starts from zero, different from source code
            'batch': step+1,  # starts from zero
            'iter': epoch * len(data_loader) + (step+1),
            'kinetics_loss': kinetics_loss.avg,
            'ucf_loss': ucf_loss.avg,
            'prec1': top1.avg.item(),
            'prec5': top5.avg.item(),
            'old_lr': optimizer.param_groups[0]['lr'],
            'new_lr': optimizer.param_groups[1]['lr']
        })

        if (step+1) % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {old_lr:.5f}/{new_lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'kinetics_loss {kinetics_loss.val:4f} ({kinetics_loss.avg:.4f})\t'
                  'ucf_loss {ucf_loss.val:.4f} ({ucf_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Prec@5 {top5.val:.4f} ({top5.avg:.4f})'.format(
                      epoch+1,
                      step+1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      kinetics_loss=kinetics_loss,
                      ucf_loss=ucf_loss,
                      top1=top1,
                      top5=top5,
                      old_lr=optimizer.param_groups[0]['lr'],
                      new_lr=optimizer.param_groups[1]['lr']
                  ))
            writer.add_scalar('Train/kinetics_loss', kinetics_loss.avg, step+1)
            writer.add_scalar('Train/ucf_loss', ucf_loss.avg, step+1)
            writer.add_scalar('Train/prec1', top1.avg.item(), step+1)
            writer.add_scalar('Train/prec5', top5.avg.item(), step+1)
    epoch_logger.log({
        'epoch': epoch+1,
        'kinetics_loss': kinetics_loss.avg,
        'ucf_loss': ucf_loss.avg,
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'old_lr': optimizer.param_groups[0]['lr'],
        'new_lr': optimizer.param_groups[1]['lr']
    })

    if (epoch+1) % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path, 'save_{}.pth'.format(epoch+1))

        states = {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)


def val_epoch(epoch, data_loader, model, criterion, opt, logger, writer):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    kinetics_loss = AverageMeter()
    ucf_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    for step, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        if not opt.no_cuda:
            targets = targets.cuda()
        with torch.no_grad():
            inputs = inputs.cuda()
            targets = targets.cuda()
            old_logits, new_logits = model(inputs)
            new_targets, old_targets = targets[:new_logits.size(0)], targets[new_logits.size(0):]

            label_loss = criterion(old_logits, old_targets)
            unlabel_loss = criterion(new_logits, new_targets)

            prec1, prec5 = calculate_accuracy(old_logits, old_targets, topk=(1, 5))

        kinetics_loss.update(unlabel_loss.item(), new_logits.size(0))
        ucf_loss.update(label_loss.item(), old_logits.size(0))
        top1.update(prec1, old_logits.size(0))
        top5.update(prec5, old_logits.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f}, ({data_time.avg:.5f})\t'
              'kinetics_loss {kinetics_loss.val:.4f}, ({kinetics_loss.avg:.4f})\t'
              'ucf_loss {ucf_loss.val:.4f}, ({ucf_loss.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                  epoch+1,
                  step+1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  kinetics_loss=kinetics_loss,
                  ucf_loss=ucf_loss,
                  top1=top1,
                  top5=top5,
              ))

    logger.log({'epoch': epoch+1,
                'kinetics_loss': kinetics_loss.avg,
                'ucf_loss': ucf_loss.avg,
                'prec1': top1.avg.item(),
                'prec5': top5.avg.item()})

    writer.add_scalar('eval/kinetics_loss', kinetics_loss.avg, epoch+1)
    writer.add_scalar('eval/ucf_loss', ucf_loss.avg, epoch+1)
    writer.add_scalar('eval/prec1', top1.avg.item(), epoch+1)
    writer.add_scalar('eval/prec5', top5.avg.item(), epoch+1)

    return top1.avg


def check_epoch(epoch, data_loader, model, criterion, opt,
                logger):
    print('train at epoch {}'.format(epoch+1))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            labels = targets['label']
            sources = targets['source']
            if not opt.no_cuda:
                labels = labels.cuda()
                inputs = inputs.cuda()

            old_logits, new_logits = model(inputs)
            _, old_targets = labels[:new_logits.size(0)], labels[new_logits.size(0):]
            old_sources = sources[new_logits.size(0):]
            print(old_sources)

            loss = criterion(old_logits, old_targets)
            prec1, prec5 = calculate_accuracy(old_logits, old_targets, topk=(1, 5))
#            outputs = model(inputs)
#            loss = criterion(outputs, targets)

#        losses.update(loss.item(), inputs.size(0))
#        prec1, prec5 = calculate_accuracy(outputs, targets, topk=(1, 5))
#        top1.update(prec1, inputs.size(0))
#        top5.update(prec5, inputs.size(0))

        losses.update(loss.item(), old_logits.size(0))
        top1.update(prec1, old_logits.size(0))
        top5.update(prec5, old_logits.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f}, ({data_time.avg:.5f})\t'
              'Loss {losses.val:.4f}, ({losses.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                  epoch+1,
                  i+1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  losses=losses,
                  top1=top1,
                  top5=top5,
              ))

    logger.log({'epoch': epoch+1,
                'loss': losses.avg,
                'prec1': top1.avg,
                'prec5': top5.avg})


def train_on_clusters(opt, epoch, data_loader, model, criterion, optimizer, epoch_logger, batch_logger, writer):
    print('Train on clusters {}, epoch {}'.format(opt.pesudo_label_file, epoch+1))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    for step, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        if not opt.no_cuda:
            targets = targets.cuda()
            inputs = inputs.cuda()

        logits = model(inputs)

        loss = criterion(logits, targets)

        losses.update(loss.item(), inputs.size(0))
        prec1, prec5 = calculate_accuracy(logits.detach(), targets.detach(), topk=(1, 5))

        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch+1,
            'batch': step+1,
            'iter': epoch * len(data_loader) + (step+1),
            'loss': losses.avg,
            'prec1': top1.avg.item(),
            'prec5': top5.avg.item(),
            'lr': optimizer.param_groups[0]['lr']
        })

        if (step+1) % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {losses.val:4f} ({losses.avg:.4f})\t'
                  'Prec@1 {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Prec@5 {top5.val:.4f} ({top5.avg:.4f})'.format(
                      epoch+1,
                      step+1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      losses=losses,
                      top1=top1,
                      top5=top5,
                      lr=optimizer.param_groups[0]['lr']
                  ))

            writer.add_scalar('Cluster/loss', losses.avg, epoch * len(data_loader) + (step + 1))
            writer.add_scalar('Cluster/prec1', top1.avg.item(), epoch * len(data_loader) + (step + 1))
            writer.add_scalar('Cluster/prec5', top5.avg.item(), epoch * len(data_loader) + (step + 1))

#        if step == 1:
#            break

    epoch_logger.log({
        'epoch': epoch+1,
        'loss': losses.avg,
        'prec1': top1.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })

#    if (epoch+1) % opt.checkpoint == 0:
#        save_file_path = os.path.join(opt.result_path, opt.store_name, 'ensemble_{}.pth'.format(epoch+1))
#
#        states = {
#            'epoch': epoch+1,
#            'state_dict': model.state_dict(),
#            'optimizer': optimizer.state_dict(),
#        }
#        torch.save(states, save_file_path)

    return losses.avg, top1.avg, top5.avg
