from .utils import choose_root


DATA_PATH = choose_root(
    '/home/dsi/shaya/cc359_data/CC359/',
    r'C:\Users\Y\PycharmProjects\cc359_data\CC359'
)


MSM_DATA_PATH = choose_root(
    '/home/dsi/shaya/multiSiteMRI/'
)
BASELINE_PATH = choose_root(
    '/home/dsi/shaya/domain_shift_anatomy/exp',
)



msm_res_dir = choose_root('/home/dsi/shaya/unsup_results_msm/')
msm_splits_dir = choose_root('/home/dsi/shaya/unsup_splits_msm/')

st_res_dir = choose_root('/home/dsi/shaya/unsup_results/')
st_splits_dir = choose_root('/home/dsi/shaya/unsup_splits/')


multiSiteMri_int_to_site = {0:'ISBI',1:"ISBI_1.5",2:'I2CVB',3:"UCL",4:"BIDMC",5:"HK" }
multiSiteMri_site_to_int = {v:k for k,v in multiSiteMri_int_to_site.items()}