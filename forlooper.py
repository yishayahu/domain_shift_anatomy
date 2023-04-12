import os
import shutil
from pathlib import Path


def print_and_run(cmd):
    print(cmd)
    os.system(cmd)
def main():
    ts =2
    for (source, target) in [(2,4),(1,2),(4,3),(2,5),(0,3)]:
        for _ in range(1):


            p = Path(f"/home/dsi/shaya/spottune_results_curriculum_bs_16/ts_size_{ts}/source_{source}_target_{target}/gradual_tl_not_keep_source_curriculum/checkpoints/")
            last_ckpt = p.parent / 'model.pth'
            if last_ckpt.exists():
                break
            best_p = p / "checkpoint__best"
            if best_p.exists():
                print("removing best p")
                shutil.rmtree(best_p)
            print(f"try {_}")
            print_and_run(f"CUDA_VISIBLE_DEVICES=6 python trainer.py --config gradual_tl_not_keep_source_curriculum --exp_name gradual_tl_not_keep_source_curriculum --device cuda:0 --source {source}  --target {target} --ts_size {ts} --base_split_dir /home/dsi/shaya/data_splits --base_res_dir /home/dsi/shaya/spottune_results_curriculum_bs_16 --batch_size 16")
if __name__ == '__main__':
    main()
