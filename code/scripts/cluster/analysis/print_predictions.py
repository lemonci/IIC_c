import argparse
import itertools
import os
import pickle
import numpy as np
import torch
from PIL import Image

import code.archs as archs
from code.utils.cluster.data import cluster_twohead_create_dataloaders
from code.utils.cluster.transforms import sobel_process

parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, default=700)
parser.add_argument("--num_imgs", type=int, default=200)
parser.add_argument("--out_root", type=str,
                    default="/home/monica/IIC_c/datasets/iid_private")

given_config = parser.parse_args()

given_config.out_dir = os.path.join(given_config.out_root,
                                    str(given_config.model_ind))

reloaded_config_path = os.path.join(given_config.out_dir, "config.pickle")
print("Loading restarting config from: %s" % reloaded_config_path)
with open(reloaded_config_path, "rb") as config_f:
  config = pickle.load(config_f)
#assert (config.model_ind == given_config.model_ind)

net = archs.__dict__[config.arch](config)
model_path = os.path.join(config.out_dir, "latest_net.pytorch")
net.load_state_dict(
  torch.load(model_path, map_location=lambda storage, loc: storage))

net.cuda()
net.eval()

net = torch.nn.DataParallel(net)

data = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder('/home/monica/Shicang/Resize_white/train'))
# iterable of tensors (image(s), label(s)), for example simple list, or dataset (e.g. ImageFolder) or DataLoader
for inputs, labels in data: # inputs dimensions: (n, c, h, w) or (c, h, w), labels dimensions: (n) or (1) but labels are ignored
  inputs = inputs.cuda() # if using GPU
  if len(inputs.shape) == 3: inputs = inputs.unsqueeze(0) # shape (c, h, w) ->  (1, c, h, w)
  inputs = sobel_process(inputs, config.include_rgb)

  with torch.no_grad(): # turn off gradients support
    predictions = net(inputs) # predictions size: (n, num_clusters) or (1, num_clusters), automatically uses head_B
    print(predictions)
print("finished rendering to: %s" % render_out_dir)
