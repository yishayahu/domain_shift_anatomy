import os
import random
import shutil
from itertools import combinations



if __name__ == '__main__':
    experiments = ['posttrain_adam', 'gradual_tl_adam', 'spottune_adam', 'posttrain_continue_optimizer', 'spottune_soft_adam']
    device = 'cuda:5'
    combs = list(combinations(list(range(6)), 2))
    random.shuffle(combs)

    for combination in combs:
        for exp in experiments:
            source, target = combination
            if source == 0 and target == 2:
                continue
            src_ckpt_path = f'/home/dsi/shaya/data_splits/sources/source_{source}/model_adam.pth'
            if not os.path.exists(src_ckpt_path):
                print(f'training on source {source}')
                os.system(f'python trainer.py --config only_source_adam --exp_name only_source_adam --device {device} --source {source} --train_only_source')
                pp_model = f'/home/dsi/shaya/spottune_results/source_{source}/only_source_adam/checkpoints/checkpoint_59/model.pth'
                os.rename(pp_model,src_ckpt_path)
                pp_optim = f'/home/dsi/shaya/spottune_results/source_{source}/only_source_adam/checkpoints/checkpoint_59/optimizer.pth'
                os.rename(pp_optim,f'/home/dsi/shaya/data_splits/sources/source_{source}/optimizer_adam.pth')


            print(f'training on source {source} target {target} exp {exp}')
            os.system(f'python trainer.py --config {exp} --exp_name {exp} --device {device} --source {source} --target {target}')
