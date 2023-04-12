from pathlib import Path

from .utils import choose_root


DATA_PATH = choose_root(
    '/home/dsi/shaya/cc359_data/CC359/'
)


MSM_DATA_PATH = choose_root(
    '/home/dsi/shaya/multiSiteMRI/'
)

BRATS_DATA_PATH = choose_root('/dsi/shared/shaya/MICCAI_BraTS_2019_Data_Training/HGG/')

BRATS_SPLITS_PATH = choose_root('/home/dsi/shaya/data_split_brats/')
BRATS_RES_DIR = choose_root('/home/dsi/shaya/brats_results/')

msm_splits_dir = choose_root('/home/dsi/shaya/data_split_msm2/')
msm_res_dir = choose_root('/home/dsi/shaya/msm_results/')

st_res_dir = Path('/home/dsi/shaya/spottune_results_curriculum/')
st_splits_dir = choose_root('/home/dsi/shaya/data_splits/')


multiSiteMri_int_to_site = {0:'ISBI',1:"ISBI_1.5",2:'I2CVB',3:"UCL",4:"BIDMC",5:"HK" }