import torch
import torch.nn as nn
import numpy as np

from collections import defaultdict, OrderedDict, deque
from ..utils.torch import to_npy
from scipy.ndimage import zoom

class Base(object):
    def __init__(self, device="cuda:0", net_names=[]):
        self.loss = defaultdict(deque)
        self.visual = OrderedDict()
        self.device = device
        self.net_names = net_names

    def _move_to_device(self, *inputs):
        if len(inputs) == 1:
            return inputs[0].to(self.device)
        else:
            return (i.to(self.device) for i in inputs)

    def _record_loss(self, name, value, max_cnt=100):
        self.loss[name].append(value.item())
        if len(self.loss[name]) > max_cnt: self.loss[name].popleft()

    def _record_visual(self, name, value, normalize=True):
        # print(value.shape, name)
        value = to_npy(value)
        if normalize:
            value = np.stack([
                ((v - v.min()) / (v.max() - v.min())) if v.max() > v.min() else v
                for v in value])
            value = value.clip(0, 1)
        value = (value * 255).astype(np.uint8)
        if value.shape[1] != 128:
            # print('reshape',value.shape)
            value = zoom(value, (1,128/value.shape[1],128/value.shape[2]))
        self.visual[name] = value

    def get_loss(self):
        return OrderedDict(sorted(
            [(k, np.mean(v)) for k, v in self.loss.items()]))

    def get_visual(self):
        name = "_".join(self.visual.keys())
        visuals = []
        for visual in self.visual.values():
            visuals.append(np.concatenate(visual))
        grid = np.concatenate(visuals, axis=1)

        return {name: grid}

    def load(self, checkpoint_file):
        print(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        for name, state_dict in checkpoint.items():
            # if 'netD' in name or 'net2d' in name:
                # continue
            # if name != 'xray_net':
            #     continue
            print(name + ' loading')
            net = getattr(self, name)
            net.load_state_dict(state_dict)
            # if 'net2d' in name:
                # net = nn.DataParallel(net)
            setattr(self, name, net)
        print(f"{checkpoint_file} loaded.")

    def save(self, checkpoint_file):
        checkpoint = {}
        for name in self.net_names:
            net = getattr(self, name)
            # checkpoint[name] = net.module.cpu().state_dict()
            checkpoint[name] = net.cpu().state_dict()
            net.to(self.device)
        torch.save(checkpoint, checkpoint_file)