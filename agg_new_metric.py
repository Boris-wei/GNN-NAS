from torch_geometric.graphgym.utils.agg_runs import (agg_runs,
                                                     agg_batch_proxy,
                                                     agg_batch,
                                                     name_to_dict,
                                                     json_to_dict_list,
                                                     rm_keys,
                                                     makedirs_rm_exist)
from torch_geometric.graphgym.config import cfg
import os
import pandas as pd


def agg_new_metric(dir, metric_best='auto'):
    results = {'train': [], 'val': [], 'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:  # TODO(wby) listdir is an arbitrary order
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                i = i + 1  # in order to give the right serial number for each combination
                for split in os.listdir(dir_run): # split = 'test' 'val' 'train'
                    dir_split = os.path.join(dir_run, split)
                    fname_stats = os.path.join(dir_split, 'best.json')
                    dict_stats = json_to_dict_list(fname_stats)[
                        -1]  # get best epoch
                    rm_keys(dict_stats,
                            ['lr', 'lr_std', 'eta', 'eta_std', 'params_std']) # remove these attribution
                    serial = {'serial': i}
                    results[split].append({**serial, **dict_name, **dict_stats}) # 将组合方式和训练结果进行组合，前面参数为组合方式，后面参数为训练结果
    dir_out = os.path.join(dir, 'agg')
    makedirs_rm_exist(dir_out)
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, '{}_best.csv'.format(key))      # 将每个combination的best.json存入csv当中
            results[key].to_csv(fname, index=False)

    results = {'train': [], 'val': [], 'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                i = i + 1
                for split in os.listdir(dir_run):
                    dir_split = os.path.join(dir_run, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    dict_stats = json_to_dict_list(fname_stats)[
                        -1]  # get last epoch
                    rm_keys(dict_stats,
                            ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])
                    serial = {'serial': i}
                    results[split].append({**serial, **dict_name, **dict_stats})
    dir_out = os.path.join(dir, 'agg')
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, '{}.csv'.format(key)) # 三次平均之后最后一个epoch
            results[key].to_csv(fname, index=False)

    results = {'train': [], 'val': [], 'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                i = i + 1
                for split in os.listdir(dir_run):
                    dir_split = os.path.join(dir_run, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    dict_stats = json_to_dict_list(
                        fname_stats)  # get best epoch
                    if metric_best == 'auto':
                        metric = 'auc' if 'auc' in dict_stats[0] \
                            else 'accuracy'
                    else:
                        metric = metric_best
                    performance_np = np.array(  # noqa
                        [stats[metric] for stats in dict_stats])
                    dict_stats = dict_stats[eval("performance_np.{}()".format(
                        cfg.metric_agg))]
                    serial = {'serial': i}
                    rm_keys(dict_stats,
                            ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])
                    results[split].append({**serial, **dict_name, **dict_stats})
    dir_out = os.path.join(dir, 'agg')
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, '{}_bestepoch.csv'.format(key)) # 三次平均之后得到的最好epoch
            results[key].to_csv(fname, index=False)

    results = {'train': [], 'val': [], 'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                i = i + 1
                split = 'val'
                dir_split = os.path.join(dir_run, split)
                fname_stats = os.path.join(dir_split, 'best.json')
                dict_stats = json_to_dict_list(fname_stats)[
                    -1]  # get best val epoch
                serial = {'serial': i}
                best_val_epoch = dict_stats['epoch']
                split = 'test'
                dir_split = os.path.join(dir_run, split)
                fname_stats = os.path.join(dir_split, 'stats.json')
                dict_stats = json_to_dict_list(fname_stats)[best_val_epoch]
                rm_keys(dict_stats,
                        ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])
                results[split].append({**serial, **dict_name, **dict_stats})
    dir_out = os.path.join(dir, 'agg')
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, 'personal_test_val_best.csv')
            results[key].to_csv(fname, index=False)

    print('Results aggregated across models saved in {}'.format(dir_out))


if __name__ == '__main__':
    new_metric = 'accuracy'
    dir = 'results/personal_graph4_grid_proxy_mod54_1'
    task = 'groundtruth'
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:
        if (run != 'agg') and (run != 'config.yaml'):
            run_dir = dir + '/' + run
            agg_runs(run_dir, metric_best=new_metric)
    if task == 'groundtruth':
        agg_batch(dir, new_metric)
    elif task == 'proxy':
        agg_batch_proxy(dir, new_metric)




