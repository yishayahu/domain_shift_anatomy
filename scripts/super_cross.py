import argparse
import itertools
import json
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import time
import numpy as np
from multiprocessing import Process

import torch.cuda
from spottunet import paths

from tqdm import tqdm
from wandb.vendor.pynvml.pynvml import nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, \
    nvmlDeviceGetUtilizationRates, nvmlInit


def find_available_device(my_devices,running_now):
    if torch.cuda.is_available():
        wanted_free_mem = 20 * 2 ** 30  # at least 16 GB avail
        while True:
            for device_num in range(nvmlDeviceGetCount()):
                if f'cuda:{device_num}' in my_devices:
                    continue
                h = nvmlDeviceGetHandleByIndex(device_num)
                info = nvmlDeviceGetMemoryInfo(h)
                gpu_utilize = nvmlDeviceGetUtilizationRates(h)
                if info.free > wanted_free_mem and gpu_utilize.gpu < 3:
                    return f'cuda:{device_num}'
            print(f'looking for device my device is {my_devices}')
            places = [x[0] for x in running_now]
            print(places)
            time.sleep(600)
    else:
        return 'cpu'

def run_single_exp(exp,device,source,target,ts,sdice_path,my_devices,ret_value,res_path,data_split_path):
    my_devices.append(device)
    print(f'training on source {source} target {target} exp {exp} on device {device} my devices is {my_devices}')
    with tempfile.NamedTemporaryFile() as out_file, tempfile.NamedTemporaryFile() as err_file:
        try:
            cmd = f'CUDA_VISIBLE_DEVICES={int(device.split(":")[1])} python trainer.py --config {exp} --exp_name {exp} --device cuda:0 --source {source} --target {target} --ts_size {ts} --base_split_dir {data_split_path} --base_res_dir {res_path} >  {out_file.name} 2> {err_file.name}'
            print(cmd)
            subprocess.run(cmd,shell=True,check=True)
            sdice = json.load(open(sdice_path))
            if type(sdice)!= float:
                sdice = np.mean(list(sdice).values())

            ret_value.value = sdice
        except subprocess.CalledProcessError:
            print(f'error in exp {exp}_{source}_{target}_{ts}')
            shutil.copy(err_file.name,f'errs_and_outs/{exp}_{source}_{target}_{ts}_logs_err.txt')
            shutil.copy(out_file.name,f'errs_and_outs/{exp}_{source}_{target}_{ts}_logs_out.txt')


    my_devices.remove(device)
def run_cross_validation(experiments, combs,data_split_path,res_path,metric,target_sizes, only_stats=False):
    if torch.cuda.is_available():
        nvmlInit()
    manager = multiprocessing.Manager()


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
                msm = '_msm' if 'msm' in exp else ''
                if target == 0 and 'msm' not in exp:
                    continue
                last_ckpt = '59' if 'msm' in exp else '59'
                src_ckpt_path = f'{data_split_path}/sources/source_{source}/model_{adam_or_sgd}.pth'

                if not os.path.exists(src_ckpt_path):
                    if only_stats:
                        continue
                    curr_device = find_available_device(my_devices,running_now)
                    print(f'training on source {source} to create {src_ckpt_path}')
                    pp_model = f'{res_path}/source_{source}/only_source_{adam_or_sgd}/checkpoints/checkpoint_{last_ckpt}/model.pth'
                    if not os.path.exists(pp_model):
                        my_devices.append(curr_device)
                        subprocess.run(f'python trainer.py --config only_source{msm}_{adam_or_sgd} --exp_name only_source_{adam_or_sgd} --device {curr_device} --source {source} --train_only_source >  errs_and_outs/only_source{source}_logs_out.txt 2> errs_and_outs/only_source{source}_logs_err.txt',shell=True,check=True)
                        my_devices.remove(curr_device)
                    os.rename(pp_model,src_ckpt_path)
                    pp_optim = f'{res_path}/source_{source}/only_source_{adam_or_sgd}/checkpoints/checkpoint_{last_ckpt}/optimizer.pth'
                    os.rename(pp_optim,f'{data_split_path}/sources/source_{source}/optimizer_{adam_or_sgd}.pth')
                sdice_path = f'{res_path}/ts_size_{ts}/source_{source}_target_{target}/{exp}/test_metrics/{metric}.json'
                if not os.path.exists(sdice_path):
                    if only_stats:
                        continue
                    curr_device = find_available_device(my_devices,running_now)
                    exp_dir_path = f'{res_path}/ts_size_{ts}/source_{source}_target_{target}/{exp}'
                    if os.path.exists(os.path.join(exp_dir_path,'.lock')):
                        print(f'source {source} target {target} exp {exp} is locked' )
                        continue
                    if os.path.exists(exp_dir_path):
                        shutil.rmtree(exp_dir_path,ignore_errors=True)
                    print(f'lunch on source {source} target {target} exp {exp}')
                    ret_value = multiprocessing.Value("d", 0.0, lock=False)
                    p = Process(target=run_single_exp,args=(exp,curr_device,source,target,ts,sdice_path,my_devices,ret_value,res_path,data_split_path))
                    running_now.append([(exp,ts,f's_{source} t_{target}'),p,ret_value])
                    p.start()
                    time.sleep(5)
                else:
                    print(f'loading exists on source {source} target {target} exp {exp}')
                    sdice = json.load(open(sdice_path))
                    if type(sdice)!= float:
                        sdice = np.mean(list(json.load(open(sdice_path)).values()))
                    stats[exp][ts][f's_{source} t_{target}'] = sdice
    still_running = running_now
    while still_running:
        still_running = []
        places = []
        for place,p,ret_value in tqdm(running_now,desc='finishing running now'):
            p.join(timeout=0)
            if p.is_alive():
                still_running.append((place,p,ret_value))
                places.append(place)

            else:
                stats[place[0]][place[1]][place[2]] = ret_value.value
        running_now = still_running
        print(places)
        if running_now:
            time.sleep(600)
    print(stats)
    json.dump(stats,open('all_stats.json','w'))
    return stats

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--msm", action='store_true')
    cli.add_argument("--st", action='store_true')
    opts = cli.parse_args()
    if opts.msm:
        experiments = ['posttrain_msm_adam','gradual_tl_msm_adam','gradual_tl__continue_optimzer_msm_adam_from_step','posttrain_continue_optimizer_from_step_msm_adam','spottune_msm_adam']
        combs = [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5)]
        metric = 'dice'
        data_split_path,res_path = paths.msm_splits_dir,paths.msm_res_dir
        run_cross_validation(only_stats=False,experiments=experiments,combs=combs,metric=metric,data_split_path=data_split_path,res_path=res_path,target_sizes=[2,4])
    if opts.st:
        base_exps_sgd = ['posttrain','spottune','posttrain_continue_optimizer','gradual_tl','unfreeze_first']
        base_exps_adam = ['posttrain_adam', 'spottune_adam', 'gradual_tl_adam','posttrain_continue_optimizer_from_step_adam','gradual_tl__continue_optimzer_adam_from_step','unfreeze_first_adam']
        experiments_base = base_exps_sgd+base_exps_adam
        experiments =  experiments_base
        combs = itertools.permutations(range(6),2)
        metric = 'sdice_score'
        data_split_path,res_path = paths.st_splits_dir,paths.st_res_dir
        run_cross_validation(only_stats=False,experiments=experiments,combs=combs,metric=metric,data_split_path=data_split_path,res_path=res_path,target_sizes=[0,1,2,4])

if __name__ == '__main__':
    main()
