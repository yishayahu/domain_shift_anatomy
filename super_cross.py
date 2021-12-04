import json
import multiprocessing
import os
import subprocess
import sys
import time
from itertools import combinations
import numpy as np
from multiprocessing import Process

import torch.cuda
from matplotlib import pyplot as plt
from tqdm import tqdm
from wandb.vendor.pynvml.pynvml import nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, \
    nvmlDeviceGetUtilizationRates, nvmlInit


def find_available_device(my_devices):
    if torch.cuda.is_available():
        wanted_free_mem = 16 * 2 ** 30  # at least 16 GB avail
        while True:
            for device_num in range(nvmlDeviceGetCount()):
                if f'cuda:{device_num}' in my_devices:
                    continue
                h = nvmlDeviceGetHandleByIndex(device_num)
                info = nvmlDeviceGetMemoryInfo(h)
                gpu_utilize = nvmlDeviceGetUtilizationRates(h)
                if info.free > wanted_free_mem and gpu_utilize.gpu < 1:
                    return f'cuda:{device_num}'
            print(f'looking for device my device is {my_devices}')
            time.sleep(600)
    else:
        return 'cpu'

def run_single_exp(exp,device,source,target,ts,sdice_path,my_devices,ret_value):
    my_devices.append(device)
    print(f'training on source {source} target {target} exp {exp} on device {device} my devices is {my_devices}')
    try:
        subprocess.run(f'python trainer.py --config {exp} --exp_name {exp} --device {device} --source {source} --target {target} --ts_size {ts} >  errs_and_outs/{exp}_{source}_{target}_logs_out.txt 2> errs_and_outs/{exp}_{source}_{target}_logs_err.txt',shell=True,check=True)
        sdice = np.mean(list(json.load(open(sdice_path)).values()))
        ret_value.value = sdice
    except subprocess.CalledProcessError:
        print(f'error in exp {exp}')


    my_devices.remove(device)
def main():
    if torch.cuda.is_available():
        nvmlInit()
    manager = multiprocessing.Manager()
    sgd_exps = ['posttrain','gradual_tl','spottune','posttrain_continue_optimizer']
    adam_exps = ['posttrain_adam', 'gradual_tl_adam', 'spottune_adam', 'posttrain_continue_optimizer_adam','spot_with_grad_adam']
    experiments = adam_exps+sgd_exps
    target_sizes = [2]#[1,2,4,8]
    combs = list(combinations(list(range(6)), 2))
    stats = {}

    running_now = []
    my_devices = manager.list()
    for combination in tqdm(combs):
        for ts in target_sizes:
            for exp in experiments:
                if exp not in stats:
                    stats[exp] = {}

                if ts not in stats[exp]:
                    stats[exp][ts] = {}
                source, target = combination
                if source == 0 and target == 2:
                    continue
                adam_or_sgd = 'adam' if 'adam' in exp else 'sgd'
                src_ckpt_path = f'/home/dsi/shaya/data_splits/sources/source_{source}/model_{adam_or_sgd}.pth'

                if not os.path.exists(src_ckpt_path):
                    curr_device = find_available_device(my_devices)
                    print(f'training on source {source} to create {src_ckpt_path}')
                    pp_model = f'/home/dsi/shaya/spottune_results/source_{source}/only_source_{adam_or_sgd}/checkpoints/checkpoint_59/model.pth'
                    if not os.path.exists(pp_model):
                        my_devices.append(curr_device)
                        subprocess.run(f'python trainer.py --config only_source_{adam_or_sgd} --exp_name only_source_{adam_or_sgd} --device {curr_device} --source {source} --train_only_source >  errs_and_outs/only_source{source}_logs_out.txt 2> errs_and_outs/only_source{source}_logs_err.txt',shell=True,check=True)
                        my_devices.remove(curr_device)
                    os.rename(pp_model,src_ckpt_path)
                    pp_optim = f'/home/dsi/shaya/spottune_results/source_{source}/only_source_{adam_or_sgd}/checkpoints/checkpoint_59/optimizer.pth'
                    os.rename(pp_optim,f'/home/dsi/shaya/data_splits/sources/source_{source}/optimizer_{adam_or_sgd}.pth')
                sdice_path = f'/home/dsi/shaya/spottune_results/ts_size_{ts}/source_{source}_target_{target}/{exp}/test_metrics/sdice_score.json'
                if not os.path.exists(sdice_path):
                    curr_device = find_available_device(my_devices)
                    print(f'lunch on source {source} target {target} exp {exp}')
                    ret_value = multiprocessing.Value("d", 0.0, lock=False)
                    p = Process(target=run_single_exp,args=(exp,curr_device,source,target,ts,sdice_path,my_devices,ret_value))
                    running_now.append([(exp,ts,f's_{source} t_{target}'),p,ret_value])
                    p.start()
                    time.sleep(5)
                else:
                    print(f'loading exists on source {source} target {target} exp {exp}')
                    sdice = np.mean(list(json.load(open(sdice_path)).values()))
                    stats[exp][ts][f's_{source} t_{target}'] = sdice
    for place,p,ret_value in tqdm(running_now,desc='finishing running now'):
        p.join()
        stats[place[0]][place[1]][place[2]] = ret_value.value
    print(stats)
    json.dump(stats,open('all_stats.json','w'))
    return stats


def plot_stats(stats):
    target_sizes = [2]
    def plot_stats_aux(sgd_or_adam):
        names = []
        means = []
        errors = []
        all1 = []
        for size in target_sizes:
            for exp_name, exp_stats in stats.items():
                if sgd_or_adam == 'adam' and 'adam' not in exp_name:
                    continue
                if sgd_or_adam == 'sgd' and 'adam'  in exp_name:
                    continue
                all1.append((exp_name.split('_')[0],np.mean(list(stats[exp_name][size].values())),np.std(list(stats[exp_name][size].values()))))

        for n,m,s in sorted(all1,key=lambda x:x[1]):
            names.append(n)
            means.append(m)
            errors.append(s)

        fig, ax = plt.subplots()
        rects = ax.bar(list(range(len(names))), means, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('sdice score')
        ax.set_xticks(list(range(len(names))))
        ax.set_xticklabels(names)
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%.3f' % height,
                    ha='center', va='bottom')
        plt.savefig(f'bar_plot_with_error_{sgd_or_adam}.png')
        plt.show()
        plt.cla()
        plt.clf()

    plot_stats_aux('sgd')
    plot_stats_aux('adam')

if __name__ == '__main__':
    plot_stats(main())
