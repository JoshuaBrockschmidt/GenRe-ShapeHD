"""
Evaluates a model against our validation split of the ShapeNet dataset.
"""

# TODO
#  * List the class and/or ID for each loss record

import csv
import os
import time
import torch
from tqdm import tqdm
from shutil import rmtree

import datasets
import models
from loggers import loggers
from options import options_test
import util.util_loadlib as loadlib
from util.util_print import str_error, str_stage, str_verbose

print("Evaluation Pipeline")


# Parse arguments
###################################################

print(str_stage, "Parsing arguments")
opt = options_test.parse()
opt.full_logdir = None
print(opt)


# Create GPU or CPU device
###################################################

print(str_stage, "Setting device")
if opt.gpu == '-1':
    device = torch.device('cpu')
else:
    loadlib.set_gpu(opt.gpu)
    device = torch.device('cuda')
if opt.manual_seed is not None:
    loadlib.set_manual_seed(opt.manual_seed)


# Create output directory
###################################################

print(str_stage, "Setting up output directory")
output_dir = opt.output_dir
output_dir += ('_' + opt.suffix.format(**vars(opt))) \
    if opt.suffix != '' else ''
opt.output_dir = output_dir

if os.path.isdir(output_dir):
    if opt.overwrite:
        rmtree(output_dir)
    else:
        raise ValueError(str_error +
                         " %s already exists, but no overwrite flag"
                         % output_dir)
os.makedirs(output_dir)


# Logger setup
###################################################

print(str_stage, "Setting up loggers")
logdir = opt.logdir
if os.path.isdir(logdir):
    if opt.overwrite:
        rmtree(logdir)
    else:
        raise ValueError(str_error +
                         " %s already exists, but no overwrite flag"
                         % logdir)
os.makedirs(logdir)
logger_list = [
    loggers.TerminateOnNaN()
]
logger = loggers.ComposeLogger(logger_list)
logger.eval()


# Recording loss
###################################################

loss_csv_path = os.path.join(output_dir, 'loss_data.csv')
if os.path.isfile(loss_csv_path):
    if opt.overwrite:
        os.remove(loss_csv_path)
    else:
        raise ValueError(str_error +
                         " %s already exists, but no overwrite flag"
                         % output_dir)
loss_file = open(loss_csv_path, 'w')
loss_csv_writer = csv.writer(loss_file, delimiter=',', quotechar='"')
header_row = ('loss', 'voxel_loss', 'surface_loss')
loss_csv_writer.writerow(header_row)


# Load model
###################################################

print(str_stage, "Setting up models")
Model = models.get_model(opt.net, test=False)
model = Model(opt, logger)
model.to(device)
model.eval()
print(model)
print("# model parameters: {:,d}".format(model.num_parameters()))


# Create data loader
###################################################

print(str_stage, "Setting up data loader")
start_time = time.time()
Dataset = datasets.get_dataset('shapenet')
dataset = Dataset(opt, mode='vali', model=model)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    pin_memory=True,
    drop_last=True,
    shuffle=False
)
n_batches = len(dataloader)
dataiter = iter(dataloader)
print(str_verbose, "Time spent in data IO initialization: %.2fs" %
      (time.time() - start_time))
print(str_verbose, "# test points: " + str(len(dataset)))
print(str_verbose, "# test batches: " + str(n_batches))


# Run evaluation
###################################################

print(str_stage, "Evaluating")
for i in tqdm(range(n_batches)):
    batch = next(dataiter)
    _, loss_data = model.eval_on_batch(i, batch)
    row = (loss_data['loss'], loss_data['voxel_loss'], loss_data['surface_loss'])
    loss_csv_writer.writerow(row)
    loss_file.flush()
close(loss_file)
