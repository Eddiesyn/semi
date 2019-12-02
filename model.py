import torch
import torch.nn as nn

from models import resnet


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_original_model(opt):
    model = resnet.resnet18(num_classes=opt.num_classes,
                            shortcut_type='A',
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    print('Use pretrained model {}'.format(opt.pretrain_path))
    pretrain = torch.load(opt.pretrain_path)
    model.load_state_dict(pretrain['state_dict'])
    model.module.fc = Identity()

    return model


def get_model(opt):
    model = resnet.resnet18(num_classes=opt.num_classes,
                            shortcut_type='A',
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)

    return model


def generate_model(opt, num_new_classes):
    model = resnet.resnet18(num_classes=opt.num_classes,
                            shortcut_type='A',
                            sample_size=opt.sample_size,
                            sample_duration=opt.sample_duration)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total number of trainable parameters {}'.format(pytorch_total_params))

        if opt.pretrain:
            assert opt.pretrain_path is not None, 'Please give the pretrained path!'
            print('loading pretrained model {}'.format(opt.pretrain_path))
#            pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
            pretrain = torch.load(opt.pretrain_path)
            model.load_state_dict(pretrain['state_dict'])
            prec1 = pretrain['best_prec1']

            old_fc = model.module.fc
            model.module.fc = nn.Linear(model.module.fc.in_features, num_new_classes)
            model.module.fc = model.module.fc.cuda()
        else:
            raise 'Use your best model'
    else:
        raise 'Buy a GPU please!'
#        if opt.pretrain:
#            assert opt.pretrain_path is not None, 'Please give the pretrained path!'
#            print('loading pretrained model {}'.format(opt.pretrain_path))
#            pretrain = torch.load(opt.pretrain_path)
#            model.load_state_dict(pretrain['state_dict'])
#            prec1 = pretrain['best_prec1']
#            model.module.fc = nn.Linear(model.module.fc.in_features, num_new_classes)

    return model, old_fc, prec1 


def generate_combined_model(opt, num_new_classes):
    old_model = resnet.resnet18(num_classes=opt.num_classes,
                                shortcut_type='A',
                                sample_size=opt.sample_size,
                                sample_duration=opt.sample_duration)
    if not opt.no_cuda:
        old_model = old_model.cuda()
        old_model = nn.DataParallel(old_model, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in old_model.parameters() if p.requires_grad)
        print('Total number of trainable parameters {}'.format(pytorch_total_params))
        if opt.pretrain:
            assert opt.pretrain_path is not None, 'Please give the pretrained path!'
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            prec1 = pretrain['best_prec1']
            old_model.load_state_dict(pretrain['state_dict'])
        else:
            raise 'Use your best model!'
    else:
        raise 'Buy a GPU please!'

    model = CombinedModel(old_model, num_new_classes, opt)
    model.new_fc = model.new_fc.cuda()

    return model, model.parameters(), prec1


class CombinedModel(nn.Module):
    def __init__(self, modelA, num_new_classes, opt):
        super(CombinedModel, self).__init__()
        self.old_model = modelA
        self.num_new_classes = num_new_classes
        self.old_fc = self.old_model.module.fc
        self.new_fc = nn.Linear(self.old_fc.in_features, self.num_new_classes)
        self.ratio = opt.ratio

    def go_seperate(self, features):
        batch_size = features.size(0)
        old_size = int(batch_size * self.ratio)
        new_size = batch_size - old_size
        old_features = features[new_size:, :]
        new_features = features[:new_size, :]
        old_logits = self.old_fc(old_features)
        new_logits = self.new_fc(new_features)

        return old_logits, new_logits

    def get_features(self, inputs):
        self.old_model.module.fc = Identity()
        return self.old_model(inputs)

    def forward(self, x, combined=True):
        # now each x contained labeled data and unlabeled data
        features = self.get_features(x)
        if combined:
            old_logits, new_logits = self.go_seperate(features)
        else:
            old_logits = self.old_fc(features)
            new_logits = None

        return old_logits, new_logits


