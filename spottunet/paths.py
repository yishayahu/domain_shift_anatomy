from .utils import choose_root


DATA_PATH = choose_root(
    '/home/dsi/shaya/cc359_data/CC359/'
)


MSM_DATA_PATH = choose_root(
    '/mnt/dsi_vol1/multiSiteMRI/'
)
BASELINE_PATH = choose_root(
    '/home/dsi/shaya/domain_shift_anatomy/exp',
)

msm_splits_dir = choose_root('/home/dsi/shaya/data_split_msm/')
msm_res_dir = choose_root('/home/dsi/shaya/msm_results/')

st_res_dir = choose_root('/home/dsi/shaya/spottune_results/')
st_splits_dir = choose_root('/home/dsi/shaya/data_splits/')


multiSiteMri_int_to_site = {0:'ISBI',1:"ISBI_1.5",2:'I2CVB',3:"UCL",4:"BIDMC",5:"HK" }