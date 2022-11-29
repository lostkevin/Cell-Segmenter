import torch
from collections import OrderedDict
from torchvision.models import resnet18, ResNet18_Weights
from utils.dataloaders import checkDictEq
CONTEST_WEIGHT = False
model = resnet18(weights=ResNet18_Weights.DEFAULT)
source_state_dict = model.state_dict()

#! downloaded resnet18-5c106cde.pth does not include num_batches_tracked, we initialize it with 0
# =================================================
# CONTEST_WEIGHT = True
# model = resnet18()
# source_state_dict = torch.load('resnet18-5c106cde.pth')
# model.load_state_dict(source_state_dict)
# source_state_dict = model.state_dict()
# =================================================

target_checkpoint = {
    'epoch': -1,
    'best_fitness': None,
    'ema': None,
    'updates': None,
    'optimizer': None,
    'data': None,
    'date': None
}

target_state_dict = OrderedDict()

for k, v in source_state_dict.items():
    if k.startswith('conv1'):
        target_state_dict['model.0.conv.' + k.split('.')[-1]] = v
    elif k.startswith('bn1'):
        target_state_dict['model.0.bn.' + k.split('.')[-1]] = v
    elif k.startswith('layer'):
        idx = int(k[5])
        target_state_dict[f'model.{idx + 1}.layer' + k[6:]] = v
    else:
        print(f'skip {k}')
    if CONTEST_WEIGHT and k.find('num_batches_tracked') != -1:
        assert v == 0

target_checkpoint['model'] = target_state_dict
torch.save(target_checkpoint, 'resnet18_converted.pt') # dump checkpoint file

if CONTEST_WEIGHT:
    target = torch.load('resnet18.pt', map_location='cpu')
    print(checkDictEq(target, target_checkpoint))