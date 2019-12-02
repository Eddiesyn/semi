import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    # general setting
    parser.add_argument('--result_path', default='./results', type=str, help='result path')
    parser.add_argument('--store_name', default='model', type=str, help='identifier of checkpoints, pth, etc')
    parser.add_argument('--checkpoint', default=10, type=int, help='Trained model is saved at every this epochs')
    parser.add_argument('--n_val_samples', default=3, type=int, help='Number of validation samples for each video')
    parser.add_argument('--mean_dataset', default='activitynet', type=str,
                        help='dataset for mean values of mean substraction (activitynet | kinetics)')
    parser.add_argument('--no_mean_norm', action='store_true', help='If true, inputs are normalized by mean')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument('--std_norm', action='store_true', help='If true, inputs are normalized by standard deviation')
    parser.set_defaults(std_norm=False)
    parser.add_argument('--norm_value', default=1, type=int,
                        help='If 1, range of inputs is [0-255], If 255, range of inputs is [0-1]')
    parser.add_argument('--learning_rate', default=0.04, type=float, help='Initial learning rate for fine tuning')
    parser.add_argument('--lr_steps', default=[40, 55, 65, 70, 200, 250], type=float,
                        nargs="+", metavar='LRSteps',
                        help='epochs to decay learning rate by 10')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler')

    # begin for video classification
    parser.add_argument('--l_vids_path', default='', type=str, help='vids path of labeled dataset (ucf101 as default)')
    parser.add_argument('--l_annotation_path', default='', type=str, help='annotation path of labeled dataset')
    parser.add_argument('--ul_vids_path', default='', type=str,
                        help='vids path of unlabeled dataset (kinetics600 as default)')
    parser.add_argument('--ul_annotation_path', default='', type=str, help='annotation path of unlabeled dataset')
    parser.add_argument('--pesudo_label_file', default='', type=str,
                        help='the clustered result used for kinetics supervision')

    pretrain = parser.add_mutually_exclusive_group()
    pretrain.add_argument('--no_pretrain', dest='pretrain', action='store_false', help='train from scratch')
    pretrain.add_argument('--pretrain', dest='pretrain', action='store_true', help='train from a pretrained path')
    pretrain.set_defaults(pretrain=True)
    parser.add_argument('--pretrain_path', default='', type=str, help='saved data of previous training')

    # cuda_flag = parser.add_mutually_exclusive_group()
    # cuda_flag.add_argument('--no_cuda', dest='cuda', action='store_false', help='use CPU')
    # cuda_flag.add_argument('--cuda', dest='cuda', action='store_true', help='use GPU')
    # parser.set_defaults(cuda=True)
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)

    parser.add_argument('--sample_size', default=112, type=int, help='input spatial size')
    parser.add_argument('--sample_duration', default=16, type=int, help='temporal duration of inputs')
    parser.add_argument('--downsample', default=1, type=int, help='Downsampling. Selecting 1 frame out of N')
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--train_crop', default='corner', type=str,
                        help='Spatial cropping method in training. random is uniform. \
                        corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--n_epochs', default=250, type=int, help='Number of total epochs to run')
    parser.add_argument('--n_threads', default=8, type=int, help='Number of workers for dataloader')

    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    # end for video classification

    # begin for semi-supervised setting
    parser.add_argument('--num_classes', default=101, type=int, help='number of classes in labeled dataset')
    parser.add_argument('--ratio', default=0.5, type=float, help='ratio of ucf samples in a combined batch')
    parser.add_argument('--features_path', default='', type=str, help='path of extracted features from ul_dataset')
    parser.add_argument('--n_samples', default=10000, type=int,
                        help='num of samples used for estimating bandwidth used for MeanShift')
    parser.add_argument('--min_bin_freq', default=100, type=int, help='used for MeanShift clustering')
    parser.add_argument('--ensemble_model', default='', type=str, help='the fine-tuned ensemble model')
    parser.add_argument('--pca_sample_size', default=1024, type=int,
                        help='sample size for pca reduction, it needs to be greater than 64')
    parser.add_argument('--resume_path', default=None, type=str, help='pth for resume training')
#    parser.add_argument('--p', type=int, default=10, help='each output select p')
#    parser.add_argument('--k', type=int, default=1000, help='total k samples for each classes')
    # end for semi-supervised setting

    args = parser.parse_args()

    return args
