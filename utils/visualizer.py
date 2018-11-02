import numpy as np
import torch
from torch import nn
from torchvision.utils import make_grid
import visdom


class Visualizer():
    def __init__(self, name, display_id=1):
        self.display_id = display_id
        self.vis = visdom.Visdom(env=name)
        self.name = name

    def throw_visdom_connection_error(self):
        print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)
    
    def reset_env(self):
        self.vis.close(env=self.name)
    
    def recover_loss(self, path):
        self.plot_data = torch.load(path)

    def plot_current_losses(self, step, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(step)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': 'loss history',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except ConnectionError:
            self.throw_visdom_connection_error()
    
    def plot_current_images(self, batch_tensor, title=None, batch_size=8, win_id=None):
        if win_id == None:
            win_id = self.display_id+1
        try:
            self.vis.images(batch_tensor, nrow=batch_size, win=win_id, opts=dict(title=title))
        except ConnectionError:
            self.throw_visdom_connection_error()
        