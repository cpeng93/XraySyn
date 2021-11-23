import os, sys
import os.path as path
from torch.utils.data import DataLoader
sys.path.insert(1, './')
from xraysyn.loaders.ct2xray_real import VolumeLoader
from xraysyn.models.ct2xray_real_gan_meta import XraySynModel
from xraysyn.utils.misc import get_config
import pickle, imageio
import numpy as np
from scipy.ndimage import rotate

opt = get_config("Test an XraySyn model", model_name="xraysyn_test", phase="train")

# Create model
model = XraySynModel(**opt.model)

dataset = VolumeLoader('test_data/')
loader = DataLoader(dataset, 1, shuffle=False)
checkpoints = [path.join(opt.run_dir, f) for f in os.listdir(opt.run_dir) if f.endswith(".pt")]
checkpoints = sorted(checkpoints, key=lambda x: path.getmtime(x))
print(checkpoints)
last_epoch = 1
if checkpoints:
    last_model = checkpoints[-1]
    last_epoch = int(last_model.split("_")[-1][:-3])
    model.load(last_model)

def to_npy(inp):
    return inp[0,0].cpu().data.numpy()

out = []
counter = 0
for data in loader:
    xray_pred, xray_refine, mat_pred, mat_refine, _,_ = model.test(data['data'])
    out.append([xray_pred, xray_refine, mat_pred, mat_refine, data['data']])
    break

os.makedirs('vis',exist_ok=True)
for i in range(21):
    imageio.imsave('vis/'+str(i)+'.png',np.rot90(out[0][1][i][0,0].cpu().data.numpy()))
# pickle.dump(out, open('test_ct2xray_real_gan_meta.pt','wb'))
