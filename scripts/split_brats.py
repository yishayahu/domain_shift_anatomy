import json
import random

from spottunet.paths import BRATS_DATA_PATH, BRATS_SPLITS_PATH

root  = BRATS_DATA_PATH

split_path = BRATS_SPLITS_PATH
sources_dir = split_path/'sources'
sources_dir.mkdir(exist_ok=True)

for ds_i , ds in enumerate(['TCIA','CBICA']):
    paths = list(root.glob(f'*{ds}*'))
    random.shuffle(paths)
    train_size = int(len(paths) * 0.85)
    test_size  = len(paths) - train_size
    ds_train_ids = list(map(lambda x:str(x.name),paths[:train_size]))
    ds_test_ids = list(map(lambda x:str(x.name),paths[train_size:]))
    source_dir_ds = sources_dir/f'source_{ds_i}'
    source_dir_ds.mkdir(exist_ok=True)
    json.dump(ds_train_ids,open(source_dir_ds / 'train_ids.json','w'))
    json.dump(ds_test_ids,open(source_dir_ds / 'test_ids.json','w'))
    json.dump(ds_test_ids,open(source_dir_ds / 'val_ids.json','w'))
    for ts in [1,2,4]:
        ts_dir = split_path / f'ts_{ts}'
        ts_dir.mkdir(exist_ok=True)
        ts_dir_ds = ts_dir/f'target_{ds_i}'
        ts_dir_ds.mkdir(exist_ok=True)
        json.dump(ds_train_ids[:ts],open(ts_dir_ds / 'train_ids.json','w'))
        json.dump(ds_test_ids,open(ts_dir_ds / 'test_ids.json','w'))
        json.dump(ds_test_ids,open(ts_dir_ds / 'val_ids.json','w'))


