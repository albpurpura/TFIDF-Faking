import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm


def load_data(fpath=''):
    if len(fpath) == 0:
        fpaths = ['data/BF_CTU.csv', 'data/BF_V.csv', 'data/BF_OU.csv']
    else:
        fpaths = fpath

    honest_data = []
    dishonest_data = []
    for fpath in fpaths:
        header = True
        for line in open(fpath):
            data = line.strip().split(',')
            if header:
                header = False
                continue
            is_honest = data[-1] == 'H'
            answers = np.array(data[:10])
            if is_honest:
                honest_data.append(answers)
            else:
                dishonest_data.append(answers)
    return np.array(honest_data), np.array(dishonest_data)


def compute_tf_idf(all_data, idfs=None):
    all_data = np.array(all_data)
    tfidfs = np.zeros_like(all_data)
    distinct_answ_values = [1, 2, 3, 4, 5]
    if idfs is None:
        idfs = np.ones(shape=(len(distinct_answ_values), all_data.shape[1])) * all_data.shape[0]
        for answ_value_idx in range(len(distinct_answ_values)):
            for answ_idx in range(all_data.shape[1]):
                unique, counts = np.unique(all_data[:, answ_idx], return_counts=True)
                idfs[answ_value_idx][answ_idx] = np.log(idfs[answ_value_idx][answ_idx] /
                                                        [counts[i] for i in range(len(counts)) if
                                                         unique[i] == distinct_answ_values[answ_value_idx]][0])

    for i in range(len(all_data)):
        for j in range(all_data.shape[1]):
            curr_v = all_data[i, j]
            curr_v_index_in_idf_matrix = \
                [k for k in range(len(distinct_answ_values)) if distinct_answ_values[k] == curr_v][0]
            tf_curr_v = np.count_nonzero(all_data[i] == curr_v)
            assert 2 == np.count_nonzero(np.array([1, 3, 3, 4]) == 3)
            assert curr_v_index_in_idf_matrix == curr_v - 1
            tfidfs[i, j] = tf_curr_v * idfs[curr_v_index_in_idf_matrix, j]
    return tfidfs, idfs


def estimate_per_question_thresholds(tfidf_scores, thr=80):
    return [np.percentile(tfidf_scores[:, j], thr) for j in range(tfidf_scores.shape[1])]


def evaluate_pair(true_lies_mask, detected_lies_mask):
    if np.sum(detected_lies_mask) > 0:
        prec = np.sum(
            np.where((detected_lies_mask == true_lies_mask) & (true_lies_mask == np.ones_like(true_lies_mask)),
                     np.ones_like(detected_lies_mask), 0)) / np.sum(detected_lies_mask)
    else:
        prec = 0.
    if np.sum(true_lies_mask) > 0:
        rec = np.sum(np.where((detected_lies_mask == true_lies_mask) & (true_lies_mask == np.ones_like(true_lies_mask)),
                              np.ones_like(detected_lies_mask), 0)) / np.sum(true_lies_mask)
    else:
        rec = 0.

    if prec + rec > 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0.0
    return prec, rec, f1


def compute_precs_recs_f1s(true_lies_mask, detected_lies_mask):
    curr_precs = []
    curr_f1s = []
    curr_recs = []
    for i in range(detected_lies_mask.shape[0]):
        prec, rec, f1 = evaluate_pair(true_lies_mask[i], detected_lies_mask[i])
        curr_f1s.append(f1)
        curr_precs.append(prec)
        curr_recs.append(rec)
    # return np.mean(curr_precs), np.mean(curr_recs), np.mean(curr_f1s)
    return curr_precs, curr_recs, curr_f1s


def optimize_thr_dist(honest_data, faked_data, true_lies_mask):
    perfs = []
    thrs = []
    for percentile in np.arange(50, 100, step=1):
        thresholds = estimate_per_question_thresholds(honest_data, percentile)
        detected_lies_mask = np.array(
            [np.where(faked_data[:, j] >= thresholds[j], np.ones_like(faked_data[:, j]), 0) for j in
             range(faked_data.shape[1])]).transpose()
        # p, r, f1 = compute_precs_recs_f1s(true_lies_mask, detected_lies_mask)
        acc = np.mean(np.where(true_lies_mask == detected_lies_mask, np.ones_like(detected_lies_mask), 0))
        perfs.append(acc)
        thrs.append(percentile)
    best_thr = thrs[np.argmax(perfs)]
    print('best thr={}'.format(best_thr))
    return best_thr


def optimize_thr(tfidfs_honest, tfidfs_faked, true_lies_mask, dname=''):
    perfs = []
    thrs = []
    out = open('./output/tfidf_perfs_at_different_percentiles_on_training_data_of_test_dataset={}.tsv'.format(dname),
               'w')
    out.write('Percentile\tAccuracy\tPrecision\tRecall\tF1 Score\n')
    for thr in np.arange(5, 100, step=5):
        thresholds = estimate_per_question_thresholds(tfidfs_honest, thr)
        detected_lies_mask = np.array(
            [np.where(tfidfs_faked[:, j] >= thresholds[j], np.ones_like(tfidfs_faked[:, j]), 0) for j in
             range(tfidfs_faked.shape[1])]).transpose()
        p, r, f1 = compute_precs_recs_f1s(true_lies_mask, detected_lies_mask)
        acc = np.mean(np.where(true_lies_mask == detected_lies_mask, np.ones_like(detected_lies_mask), 0))
        perfs.append(np.mean(p))
        thrs.append(thr)
        out.write('{}\t{}\t{}\t{}\t{}\n'.format(thr, acc, np.mean(p), np.mean(r), np.mean(f1)))
    best_thr = thrs[np.argmax(perfs)]
    out.close()
    print('best percentile={}'.format(best_thr))
    return best_thr


def get_train_test_data(test_set):
    all_fpaths = ['data/BF_CTU.csv', 'data/BF_V.csv', 'data/BF_OU.csv']
    all_fpaths.remove(test_set)
    assert len(all_fpaths) == 2
    hdata_train, ldata_train = load_data(all_fpaths)
    hdata_train = np.array(hdata_train, dtype=np.float)
    ldata_train = np.array(ldata_train, dtype=np.float)

    hdata_test, ldata_test = load_data([test_set])
    hdata_test = np.array(hdata_test, dtype=np.float)
    ldata_test = np.array(ldata_test, dtype=np.float)
    return hdata_train, hdata_test, ldata_train, ldata_test


def compute_perf_distr_model(test_set):
    print('Distribution Model')
    hdata_train, hdata_test, ldata_train, ldata_test = get_train_test_data(test_set)
    true_lies_mask_train = np.where(hdata_train != ldata_train, np.ones_like(hdata_train), 0)
    true_lies_mask_test = np.where(hdata_test != ldata_test, np.ones_like(ldata_test), 0)
    percentile = optimize_thr_dist(hdata_train, ldata_train, true_lies_mask_train)
    print('percentile: {}'.format(percentile))
    thresholds = estimate_per_question_thresholds(hdata_train, percentile)
    detected_lies_mask_test = np.array(
        [np.where(hdata_test[:, j] >= thresholds[j], np.ones_like(hdata_test[:, j]), 0) for j in
         range(hdata_test.shape[1])]).transpose()
    p, r, f1 = compute_precs_recs_f1s(true_lies_mask_test, detected_lies_mask_test)
    accs = np.mean(np.where(true_lies_mask_test == detected_lies_mask_test, np.ones_like(true_lies_mask_test),
                            np.zeros_like(true_lies_mask_test)), axis=-1)
    p = np.mean(p)
    r = np.mean(r)
    f1 = np.mean(f1)
    print('Prec: {}, Recall: {}, F1Score: {}, Accuracy: {}'.format(p, r, f1, np.mean(accs)))
    print('{:.4f} & {:.4f} & {:.4f} & {:.4f}'.format(p, r, f1, np.mean(accs)))


def get_k_closest_vecs(v, pool, k):
    dists = [np.sum(np.dot(v, p) / (np.linalg.norm(v, ord=2) * np.linalg.norm(p, ord=2))) for p in pool]
    return np.array(pool)[np.argsort(dists)[: k]]


def compute_detected_lies_mask(v, closest_vecs, thr=1.0):
    avg_neighbor = np.mean(closest_vecs, axis=0)
    assert avg_neighbor.shape == closest_vecs[0].shape
    return np.where(np.abs(avg_neighbor - v) > thr, np.ones_like(v), 0)


def compute_perf_tfidf(test_set):
    print('TFIDF MODEL')
    fnames_mapper = {'data/BF_CTU.csv': 'C', 'data/BF_V.csv': 'S', 'data/BF_OU.csv': 'H'}
    hdata_train, hdata_test, ldata_train, ldata_test = get_train_test_data(test_set)
    true_lies_mask_train = np.where(hdata_train != ldata_train, np.ones_like(hdata_train), 0)
    true_lies_mask_test = np.where(hdata_test != ldata_test, np.ones_like(ldata_test), 0)

    tfidfs_honest_train, idfs_honest = compute_tf_idf(hdata_train)
    tfidfs_faked_train, _ = compute_tf_idf(ldata_train, idfs_honest)
    tfidfs_faked_test, _ = compute_tf_idf(ldata_test, idfs_honest)
    tfidfs_honest_test, _ = compute_tf_idf(hdata_test, idfs_honest)

    thr = optimize_thr(tfidfs_honest_train, tfidfs_faked_train, true_lies_mask_train, dname=fnames_mapper[test_set])
    thresholds = estimate_per_question_thresholds(tfidfs_honest_train, thr)
    # thresholds = optimize_thr_by_item(tfidfs_honest_train, tfidfs_faked_train, true_lies_mask_train)
    detected_lies_mask_test = np.array(
        [np.where(tfidfs_faked_test[:, j] >= thresholds[j], np.ones_like(tfidfs_faked_test[:, j]), 0) for j in
         range(tfidfs_faked_test.shape[1])]).transpose()
    assert detected_lies_mask_test.shape == true_lies_mask_test.shape
    accs = np.mean(np.where(true_lies_mask_test == detected_lies_mask_test, np.ones_like(true_lies_mask_test),
                            np.zeros_like(true_lies_mask_test)), axis=-1)
    precs = []
    recs = []
    f1s = []
    for i in range(detected_lies_mask_test.shape[0]):
        prec, rec, f1 = evaluate_pair(true_lies_mask_test[i], detected_lies_mask_test[i])
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    print(
        'Multi-label classification task: Precision: {}, Recall: {}, F1 Score: {}, Accuracy: {}'.format(np.mean(precs),
                                                                                                        np.mean(recs),
                                                                                                        np.mean(f1s),
                                                                                                        np.mean(accs)))
    print('{:.4f} & {:.4f} & {:.4f} & {:.4f}'.format(np.mean(precs), np.mean(recs), np.mean(f1s), np.mean(accs)))

    for index in tqdm(range(len(tfidfs_honest_test)), 'creating boxplots'):
        create_boxplot(tfidfs_honest_test, tfidfs_honest_test[index], tfidfs_faked_test[index],
                       true_lies_mask_test[index],
                       opath='./output/figures/tfidf/boxplots',
                       fname=fnames_mapper[test_set] + '_box_plot_idx={}'.format(index))

    per_question_accuracy = compute_per_question_accuracy(true_lies_mask_test, detected_lies_mask_test)

    create_perf_histogram(per_question_accuracy, fname=fnames_mapper[test_set], measure_name='Accuracy')

    # create_accuracy_hist_by_n_lies(true_lies_mask_test, detected_lies_mask_test, fname=fnames_mapper[test_set])
    create_hist_by_n_lies(true_lies_mask_test, accs, fname=fnames_mapper[test_set] + '_acc',
                          measure_name='Accuracy',
                          opath='./output/figures/tfidf/')
    create_hist_by_n_lies(true_lies_mask_test, precs, fname=fnames_mapper[test_set] + '_prec',
                          measure_name='Precision',
                          opath='./output/figures/tfidf/')
    create_hist_by_n_lies(true_lies_mask_test, recs, fname=fnames_mapper[test_set] + '_rec',
                          measure_name='Recall',
                          opath='./output/figures/tfidf/')
    create_hist_by_n_lies(true_lies_mask_test, f1s, fname=fnames_mapper[test_set] + '_f1',
                          measure_name='F1 Score',
                          opath='./output/figures/tfidf/')


def create_boxplot(honest_resps, curr_honest_resp, curr_faker_resp, true_lies_mask, opath, fname):
    if not os.path.exists(opath):
        os.makedirs(opath)
    faked_answ_indices = np.array([i for i in range(len(true_lies_mask)) if true_lies_mask[i] > 0])
    if len(faked_answ_indices) > 0:
        _ = plt.figure()
        plt.rcParams.update({'font.size': 20, 'legend.fontsize': 20})
        _ = plt.figure(figsize=(15, 10))
        _ = plt.plot([i + 1 for i in range(len(curr_faker_resp))], curr_faker_resp)
        _ = plt.scatter([i + 1 for i in faked_answ_indices], np.array(curr_faker_resp)[faked_answ_indices], color='r',
                        s=100)
        _ = plt.boxplot(honest_resps, labels=[f'Q{i + 1}' for i in range(len(curr_honest_resp))], showmeans=True,
                        whis=0.75)
        # _ = plt.show()
        plt.savefig(os.path.join(opath, fname), bbox_inches='tight')
        plt.close()


def compute_faked_answ_mask(pred_n_lies_per_test, tfidfs_faked_test, thresholds):
    new_lies_masks = []
    for i in range(tfidfs_faked_test.shape[0]):
        n_pred_lies = pred_n_lies_per_test[i]
        lies_indices = np.argsort(-np.where(tfidfs_faked_test[i] >= thresholds, tfidfs_faked_test[i], 0))[
                       :int(np.ceil(n_pred_lies))]
        curr_lies_mask = np.zeros_like(tfidfs_faked_test[i])
        for k in lies_indices:
            curr_lies_mask[k] = 1
        new_lies_masks.append(curr_lies_mask)
    return np.array(new_lies_masks)


def compute_per_question_accuracy(true_lies_mask, detected_lies_mask):
    per_question_accuracy = []
    for question_index in range(true_lies_mask.shape[1]):
        accuracy = np.mean(np.where(true_lies_mask[:, question_index] == detected_lies_mask[:, question_index],
                                    np.ones_like(detected_lies_mask[:, question_index]), 0))
        per_question_accuracy.append(accuracy)

    return np.array(per_question_accuracy)


def compute_n_lies_per_sample_dist(honest, dishonest):
    true_lies_mask = np.where(np.abs(honest - dishonest) > 0, np.ones_like(honest), 0)
    n_lies = np.sum(true_lies_mask, axis=-1)
    import collections
    counter = collections.Counter(n_lies)
    return counter


def is_stat_significant_diff(series1, series2):
    tstat, pvalue = stats.ttest_rel(series1, series2, nan_policy='omit')
    stat_sig = np.where(pvalue < 0.05, True, False)
    return stat_sig


def create_hist_by_n_lies(true_lies_mask, per_test_perf, measure_name, fname, opath='./output/figures/tfidf/'):
    if not os.path.exists(opath):
        os.makedirs(opath)
    # n_lies_map = collections.Counter(np.sum(true_lies_mask, axis=-1))
    n_lies_map = {}
    for i in range(true_lies_mask.shape[0]):
        key = np.sum(true_lies_mask[i])
        if key not in n_lies_map.keys():
            n_lies_map[key] = [i]
        else:
            n_lies_map[key].append(i)

    # per_test_perf = []
    # for test_index in range(true_lies_mask.shape[0]):
    #     accuracy = np.mean(np.where(true_lies_mask[test_index] == detected_lies_mask[test_index],
    #                                 np.ones_like(detected_lies_mask[test_index]), 0))
    #     per_test_accuracy.append(accuracy)
    per_test_perf = np.array(per_test_perf)
    y = []
    x = []
    gsize = []
    n_lies_all = sorted(list(n_lies_map.keys()))
    for n_lies in n_lies_all:
        if n_lies > 0:
            indices = n_lies_map[n_lies]
            group_acc = np.mean(per_test_perf[indices])
            x.append(n_lies)
            y.append(group_acc)
            gsize.append(len(indices) / len(true_lies_mask))

    # gsize = np.array(gsize)[np.argsort(x)]
    # y = np.array(y)[np.argsort(x)]
    # x = np.array(x)[np.argsort(x)]

    plt.figure()
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 20})
    ax = plt.gca()
    # plt.title('Faking detection {} by number of faked answers'.format(measure_name))
    plt.bar(x=x, height=y, label='{}'.format(measure_name))
    plt.xlabel('Number of faked answers per sample', fontsize=20)
    plt.ylabel('{}'.format(measure_name), fontsize=20)

    plt.plot(sorted(list(x)), gsize, color='r', label='Group sizes PDF')
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylim(0, 1)
    plt.xlim(0.5, max(x) + 1)
    leg = ax.legend(prop={'size': 20})
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=5)
    plt.grid(True)
    plt.savefig(opath + fname + '-{}-by-n-lies.png'.format(measure_name), bbox_inches='tight',
                pad_inches=0.01)


def create_perf_histogram(avgd_perfs_by_question, fname, measure_name, opath='./output/figures/tfidf/'):
    if not os.path.exists(opath):
        os.makedirs(opath)
    x = [i + 1 for i in range(len(avgd_perfs_by_question))]

    plt.figure()
    ax = plt.gca()
    plt.title('Faking detection {}'.format(measure_name))
    plt.bar(x=x, height=avgd_perfs_by_question, label=measure_name)
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)
    plt.savefig(opath + fname + '.png')


def run():
    for test_set in ['data/BF_CTU.csv', 'data/BF_V.csv', 'data/BF_OU.csv']:
        print(test_set)
        compute_perf_tfidf(test_set)
        # compute_perf_distr_model(test_set)


if __name__ == '__main__':
    run()
