import argparse
import json
import os
import torch
import tqdm

#=====START: ADDED FOR DISTRIBUTED======
from LibriDataset import LibriDataset
from waveglow_distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======

from torch.utils.data import DataLoader
from glow import WaveGlow, WaveGlowLoss
from mel2samp import Mel2Samp
import datasets

output_directory = 'valdata'

with open('config.json') as f:
    data = f.read()
config = json.loads(data)
train_config = config["train_config"]
global data_config
data_config = config["data_config"]

data_config['hf_ds'] = datasets.load_dataset('librispeech_asr', 'clean', split='validation')
del data_config['training_files']
trainset = LibriDataset(**data_config)

if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
    os.chmod(output_directory, 0o775)
print("output directory", output_directory)

from tensorboardX import SummaryWriter
logger = SummaryWriter(os.path.join(output_directory, 'logs'))


sigma = train_config['sigma']
num_gpus = 1
batch_size = train_config['batch_size']

criterion = WaveGlowLoss(sigma)

train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None

train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)

import glob, natsort, math

with open('vallosses.log', 'a') as f:
    for filename in (outer_loop := tqdm.tqdm(natsort.natsorted(glob.glob('checkpoints/waveglow_*'))[6:])):
        iteration = int(filename.split('_')[1])
        outer_loop.set_description(f"Iteration {iteration}")
        losses = []

        model = torch.load(filename)['model'].eval().cuda()

        for i, batch in enumerate(inner_loop := tqdm.tqdm(train_loader)):
            mel, audio = batch
            mel = torch.autograd.Variable(mel.cuda())
            audio = torch.autograd.Variable(audio.cuda())
            outputs = model((mel, audio))

            loss = criterion(outputs)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            inner_loop.set_description(f"Loss: {reduced_loss}")

            losses.append(reduced_loss)

        last_loss = math.fsum(losses) / len(losses)
        print(f"Iteration {iteration}: ", last_loss, file=f)
        f.flush()
        logger.add_scalar('training_loss', last_loss, iteration)

        del model
        torch.cuda.empty_cache()
