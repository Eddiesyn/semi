from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import os
import time
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--features_path', default='', type=str, help='the features waited to be clustered')
    parser.add_argument('--n_samples', default=10000, type=int, help='the number of samples used to estimate bandwidth')
    parser.add_argument('--min_bin_freq', default=100, type=int,
                        help='hyperparameters for MeanShift algorithm speed up, see details in sklearn')
#    parser.add_argument('--dst_path', default='', type=str, help='dst path of saved cluster labels')

    args = parser.parse_args()

    return args


def deep_cluster(features_path, n_samples, min_bin_freq):
    begin_time = time.time()
    print('Using features in {}'.format(features_path))
    basename = os.path.basename(features_path)
    store_folder = os.path.splitext(basename)[0]
    current_folder = os.path.dirname(features_path)
    if not os.path.exists(os.path.join(current_folder, store_folder)):
        os.makedirs(os.path.join(current_folder, store_folder))
    print('\tSaving results in {}'.format(os.path.join(current_folder, store_folder)))
    dst_path = os.path.join(current_folder, store_folder)

    big_array = np.load(features_path)
    print('retrieve {} {}-dim features done in {:.5f}s'.format(big_array.shape[0],
                                                               big_array.shape[1],
                                                               time.time()-begin_time))

    begin_time = time.time()
    bandwidth = estimate_bandwidth(big_array, n_samples=n_samples, n_jobs=-1)
    print('bandwidth {:.3f} found in {:.5f}s using {} samples'.format(bandwidth,
                                                                      time.time() - begin_time,
                                                                      n_samples))
#    import pdb; pdb.set_trace()
    begin_time = time.time()
    print('------------------')
    """
    Here base and end value need careful trying, since too big or too small bandwidth
    may yield no cluster
    """
#    base = bandwidth / 4 + 0.5
    base = bandwidth / 4
#    end = bandwidth / 3 - 0.1
#    base = bandwidth / 3
    end = bandwidth
    for bw in np.arange(base, end+0.1, 0.1):
        try:
            print('Try cluster with bandwidth {}'.format(bw))
            clustering = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=min_bin_freq, n_jobs=-1).fit(big_array)
        except ValueError as e:
            print(e)
            continue

        labels = clustering.labels_
        if not len(np.unique(labels)) > 1:
            # iteration should stop since larger bandwidth will not cluster from now on
            print('bandwidth {} failed to cluster'.format(bw))
#            break
        else:
            print('clustering done in {:.5f}s, found {} clusters'.format(time.time()-begin_time,
                                                                         len(np.unique(labels))))

            np.save(os.path.join(dst_path, 'bandwidth_{:.3f}_min{}.npy'.format(bw, min_bin_freq)), labels)
            begin_time = time.time()


if __name__ == '__main__':
    args = get_args()
    deep_cluster(args.features_path,
                 args.n_samples,
                 args.min_bin_freq)
