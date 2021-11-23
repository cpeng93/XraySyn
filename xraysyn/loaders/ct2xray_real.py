import numpy as np
import torch

from ..utils.misc import read_dir
from ..utils.geometry import get_6dofs_transformation_matrix
from torch.utils.data import Dataset
import imageio
from skimage.transform import rotate

class VolumeLoader(Dataset):
    def __init__(self, data_dir):
        super(VolumeLoader, self).__init__()
        self.style_files = read_dir(data_dir, lambda x: x.endswith("png"))
        self.device = torch.device('cuda')
        self.param_bound_other = [0.05, 0.05, 0, 0, 0, 0]
        self.param_bound_in = [0.025, 0.025, 0, 0, 0, 0]
    def to_cpu(self, x):
        return x[0].cpu()

    def norm(self, inp):
        inp = inp - inp.min()
        return inp/inp.max()        

    def get_random_param(self, flag_in):
        param = np.random.rand(6)
        if flag_in:
            param = param * self.param_bound_in * 2 - self.param_bound_in
        else:
            param = param * self.param_bound_other * 2 - self.param_bound_other
        param[0] = param[0] + 1
        param = param*np.pi
        T_in = get_6dofs_transformation_matrix(param[3:], param[:3])
        return torch.FloatTensor(T_in).to(self.device)

    def get_item(self, index):
        style_file = self.style_files[index]
        print(style_file)
        style = imageio.imread(style_file).astype(float)[...,0]
        style = rotate(style, 270)
        T_in = self.get_random_param(True)
        T_other = self.get_random_param(False)
        style = torch.FloatTensor(style/255).to(self.device).view(1,256,256)
        return {'data':self.norm(style), 'T_other': T_other, 'T_in': T_in}

    def __len__(self): return len(self.style_files)

    def __getitem__(self, index): return self.get_item(index)

