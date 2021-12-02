import json
import multiprocessing
import os
import sys
import time
from itertools import combinations
import numpy as np
from multiprocessing import Process

from tqdm import tqdm
from wandb.vendor.pynvml.pynvml import nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, \
    nvmlDeviceGetUtilizationRates, nvmlInit


def find_available_device():
    wanted_free_mem = 16 * 2 ** 30  # at least 16 GB avail
    while True:
        for device_num in range(nvmlDeviceGetCount()):
            h = nvmlDeviceGetHandleByIndex(device_num)
            info = nvmlDeviceGetMemoryInfo(h)
            gpu_utilize = nvmlDeviceGetUtilizationRates(h)
            if info.free > wanted_free_mem and gpu_utilize.gpu < 1:
                return device_num
        time.sleep(60)
        print('looking for device')

def run_single_exp(exp,device,source,target,ts,sdice_path,stats):
    print(f'training on source {source} target {target} exp {exp}')
    sys.stdout = open(os.devnull, 'w')
    os.system(f'python trainer.py --config {exp} --exp_name {exp} --device {device} --source {source} --target {target} --ts_size {ts}')
    sdice = np.mean(list(json.load(open(sdice_path)).values()))
    stats[exp][ts][f's_{source} t_{target}'] = sdice
def main():
    nvmlInit()
    manager = multiprocessing.Manager()
    sgd_exps = ['posttrain','gradual_tl','spottune','posttrain_continue_optimizer']
    adam_exps = ['posttrain_adam', 'gradual_tl_adam', 'spottune_adam', 'posttrain_continue_optimizer_adam','spot_with_grad_adam']
    experiments = adam_exps+sgd_exps
    target_sizes = [2]#[1,2,4,8]
    combs = list(combinations(list(range(6)), 2))
    stats = manager.dict()
    running_now = []
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
                    print(f'training on source {source}')
                    os.system(f'python trainer.py --config only_source_{adam_or_sgd} --exp_name only_source_{adam_or_sgd} --device {find_available_device()} --source {source} --train_only_source')
                    pp_model = f'/home/dsi/shaya/spottune_results/source_{source}/only_source_{adam_or_sgd}/checkpoints/checkpoint_59/model.pth'
                    os.rename(pp_model,src_ckpt_path)
                    pp_optim = f'/home/dsi/shaya/spottune_results/source_{source}/only_source_{adam_or_sgd}/checkpoints/checkpoint_59/optimizer.pth'
                    os.rename(pp_optim,f'/home/dsi/shaya/data_splits/sources/source_{source}/optimizer_{adam_or_sgd}.pth')
                sdice_path = f'/home/dsi/shaya/spottune_results/ts_size_{ts}/source_{source}_target_{target}/{exp}/test_metrics/sdice_score.json'
                if not os.path.exists(sdice_path):
                    p = Process(target=run_single_exp,args=(exp,find_available_device(),source,target,ts,sdice_path,stats))
                    running_now.append(p)
                    p.start()
                else:
                    sdice = np.mean(list(json.load(open(sdice_path)).values()))
                    stats[exp][ts][f's_{source} t_{target}'] = sdice
    for p in running_now:
        p.join()
    print(stats)
    json.dump(stats,open('all_stats.json','w'))

if __name__ == '__main__':
    main()