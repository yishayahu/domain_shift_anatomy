

import nibabel as nib
from torchvision import transforms
import numpy as np
import torch

from spottunet.brats2019.base_dataset import BaseDataset
from spottunet.brats2019.data_aug_3d import  ColorJitter3D, PadIfNecessary, SpatialRotation, SpatialFlip, getBetterOrientation, toGrayScale
from spottunet.paths import BRATS_DATA_PATH


class brain3DDataset(BaseDataset):
    def __init__(self,train_ids):
        super().__init__()

        # self.A1_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 't1', opt.phase), 't1.nii.gz'))
        # self.A2_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'flair', opt.phase), 'flair.nii.gz'))
        # self.B_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'dir', opt.phase), 'dir.nii.gz'))

        self.A_paths = [x.name for x in BRATS_DATA_PATH.glob('*')]
        self.train_ids = set(train_ids)
        transformations = [
            transforms.Lambda(lambda x: getBetterOrientation(x, "IPL")),
            transforms.Lambda(lambda x: np.array(x.get_fdata())[np.newaxis, ...]),
            # transforms.Lambda(lambda x: sk_trans.resize(x, (256, 256, 160), order = 1, preserve_range=True)),
            # image size [1, 160, 240, 240]
            transforms.Lambda(lambda x: x[:,8:152,24:216,24:216]),
            # transforms.Lambda(lambda x: resize(x, (x.shape[0],), order=1, anti_aliasing=True)),
            transforms.Lambda(lambda x: toGrayScale(x)),
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            # transforms.Resize((256, 256)),
            PadIfNecessary(3),
        ]

        self.updateTransformations += [
            SpatialRotation([(1,2), (1,3), (2,3)], [*[0]*12,1,2,3], auto_update=False), # With a probability of approx. 51% no rotation is performed
            SpatialFlip(dims=(1,2,3), auto_update=False)
        ]
        # transformations += self.updateTransformations
        self.transform = transforms.Compose(transformations)
        self.train_transform = transforms.Compose(self.updateTransformations)
        self.colorJitter = ColorJitter3D((0.3,1.5), (0.3,1.5))


    def load_image(self,id1):
        return self.__getitem__(self.A_paths.index(id1),only_seg=False,is_train=id1 in self.train_ids)[0]

    def load_segm(self,id1):
        return self.__getitem__(self.A_paths.index(id1),only_seg=True)[1]

    @staticmethod
    def glob_path(item,pattern):
        res = list(item.glob(pattern))
        assert len(res) == 1
        return str(res[0])

    def __getitem__(self, index,only_seg=False,is_train=False):
        x= None
        item_path = BRATS_DATA_PATH/self.A_paths[index]
        t2_path = self.glob_path(item_path,'*t2.nii')
        flair_path = self.glob_path(item_path,'*flair.nii')
        seg_path = self.glob_path(item_path,'*seg.nii')
        if not only_seg:

            t2_img = nib.load(t2_path)
            flair_img = nib.load(flair_path)
            t2_img = self.transform(t2_img)
            flair_img = self.transform(flair_img)
            if is_train:
                t2_img = self.train_transform(t2_img)
                flair_img = self.train_transform(flair_img)
                t2_img = self.colorJitter(t2_img)
                flair_img = self.colorJitter(flair_img, no_update=True)
            x = torch.concat((t2_img, flair_img), dim=0)

        y = nib.load(seg_path)
        y = self.transform(y).squeeze()
        y = y.type(torch.LongTensor)
        return x,y

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.A_paths)