import matplotlib.pylab as plt
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class Swish(nn.Module):
    
    def forward(self, x):
        return x * torch.sigmoid(x)

class Flatten(nn.Module):
    
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class SqueezeExcitation(nn.Module):
    
    def __init__(self, inplanes, se_planes):
        super(SqueezeExcitation, self).__init__()
        self.reduce_expand = nn.Sequential(
            nn.Conv2d(inplanes, se_planes, 
                      kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(se_planes, inplanes, 
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_se = torch.mean(x, dim=(-2, -1), keepdim=True)
        x_se = self.reduce_expand(x_se)
        return x_se * x

class MBConv(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, stride, 
                 expand_rate=1.0, se_rate=0.25, 
                 drop_connect_rate=0.2):
        super(MBConv, self).__init__()

        expand_planes = int(inplanes * expand_rate)
        se_planes = max(1, int(inplanes * se_rate))

        self.expansion_conv = None        
        if expand_rate > 1.0:
            self.expansion_conv = nn.Sequential(
                nn.Conv2d(inplanes, expand_planes, 
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
                Swish()
            )
            inplanes = expand_planes

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(inplanes, expand_planes,
                      kernel_size=kernel_size, stride=stride, 
                      padding=kernel_size // 2, groups=expand_planes,
                      bias=False),
            nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
            Swish()
        )

        self.squeeze_excitation = SqueezeExcitation(expand_planes, se_planes)
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(expand_planes, planes, 
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes, momentum=0.01, eps=1e-3),
        )

        self.with_skip = stride == 1
        self.drop_connect_rate = drop_connect_rate
    
    def _drop_connect(self, x):        
        keep_prob = 1.0 - self.drop_connect_rate
        drop_mask = torch.rand(x.shape[0], 1, 1, 1) + keep_prob
        drop_mask = drop_mask.type_as(x)
        drop_mask.floor_()
        return drop_mask * x / keep_prob
        
    def forward(self, x):
        z = x
        if self.expansion_conv is not None:
            x = self.expansion_conv(x)

        x = self.depthwise_conv(x)
        x = self.squeeze_excitation(x)
        x = self.project_conv(x)
        
        # Add identity skip
        if x.shape == z.shape and self.with_skip:            
            if self.training and self.drop_connect_rate is not None:
                x = self._drop_connect(x)
            x += z
        return x

def init_weights(module):    
    if isinstance(module, nn.Conv2d):    
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out')
    elif isinstance(module, nn.Linear):
        init_range = 1.0 / math.sqrt(module.weight.shape[1])
        nn.init.uniform_(module.weight, a=-init_range, b=init_range)
        
class EfficientNet(nn.Module):
        
    def _setup_repeats(self, num_repeats):
        return int(math.ceil(self.depth_coefficient * num_repeats))
    
    def _setup_channels(self, num_channels):
        num_channels *= self.width_coefficient
        new_num_channels = math.floor(num_channels / self.divisor + 0.5) * self.divisor
        new_num_channels = max(self.divisor, new_num_channels)
        if new_num_channels < 0.9 * num_channels:
            new_num_channels += self.divisor
        return new_num_channels

    def __init__(self, num_classes=100, 
                 width_coefficient=1.0,
                 depth_coefficient=1.0,
                 se_rate=0.25,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2):
        super(EfficientNet, self).__init__()
        
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.divisor = 8
                
        list_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        list_channels = [self._setup_channels(c) for c in list_channels]
                
        list_num_repeats = [1, 2, 2, 3, 3, 4, 1]
        list_num_repeats = [self._setup_repeats(r) for r in list_num_repeats]        
        
        expand_rates = [1, 6, 6, 6, 6, 6, 6]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]

        # Define stem:
        self.stem = nn.Sequential(
            nn.Conv2d(3, list_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(list_channels[0], momentum=0.01, eps=1e-3),
            Swish()
        )
        
        # Define MBConv blocks
        blocks = []
        counter = 0
        num_blocks = sum(list_num_repeats)
        for idx in range(7):
            num_channels = list_channels[idx]
            next_num_channels = list_channels[idx + 1]
            num_repeats = list_num_repeats[idx]
            expand_rate = expand_rates[idx]
            kernel_size = kernel_sizes[idx]
            stride = strides[idx]
            drop_rate = drop_connect_rate * counter / num_blocks
            
            name = "MBConv{}_{}".format(expand_rate, counter)
            blocks.append((
                name,
                MBConv(num_channels, next_num_channels, 
                       kernel_size=kernel_size, stride=stride, expand_rate=expand_rate, 
                       se_rate=se_rate, drop_connect_rate=drop_rate)
            ))
            counter += 1
            for i in range(1, num_repeats):                
                name = "MBConv{}_{}".format(expand_rate, counter)
                drop_rate = drop_connect_rate * counter / num_blocks                
                blocks.append((
                    name,
                    MBConv(next_num_channels, next_num_channels, 
                           kernel_size=kernel_size, stride=1, expand_rate=expand_rate, 
                           se_rate=se_rate, drop_connect_rate=drop_rate)                                    
                ))
                counter += 1
        
        self.blocks = nn.Sequential(OrderedDict(blocks))
        
        # Define head
        self.head = nn.Sequential(
            nn.Conv2d(list_channels[-2], list_channels[-1], 
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(list_channels[-1], momentum=0.01, eps=1e-3),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(list_channels[-1], num_classes)
        )

        self.apply(init_weights)
        
    def forward(self, x):
        f = self.stem(x)
        f = self.blocks(f)
        y = self.head(f)
        return y

model = EfficientNet(num_classes=1000, 
                     width_coefficient=1.0, depth_coefficient=1.0, 
                     dropout_rate=0.2)

!mkdir /tmp/efficientnet_weights
!wget http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth -O/tmp/efficientnet_weights/efficientnet-b0-08094119.pth

model_state = torch.load("/tmp/efficientnet_weights/efficientnet-b0-08094119.pth")

# A basic remapping is required
mapping = {
    k: v for k, v in zip(model_state.keys(), model.state_dict().keys())
}
mapped_model_state = OrderedDict([
    (mapping[k], v) for k, v in model_state.items()
])

model.load_state_dict(mapped_model_state, strict=False)

import json

with open("/tmp/labels_map.txt", "r") as h:
    labels = json.load(h)

from PIL import Image
import torchvision.transforms as transforms


img = Image.open("/tmp/giant_panda.jpg")
# Preprocess image
image_size = 224
tfms = transforms.Compose([transforms.Resize(image_size), 
                           transforms.CenterCrop(image_size), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
x = tfms(img).unsqueeze(0)

plt.imshow(img)

# Classify
model.eval()
with torch.no_grad():
    y_pred = model(x)

# Print predictions
print('-----')
for idx in torch.topk(y_pred, k=5).indices.squeeze(0).tolist():
    prob = torch.softmax(y_pred, dim=1)[0, idx].item()
    print('{label:<75} ({p:.2f}%)'.format(label=labels[str(idx)], p=prob*100))


# Dataflow:
#     setup train/test image transforms
#     setup train/test data loaders

# According to the paper authors borrowed training settings from other publications and the dataflow for CIFAR100 is the following:

#     input images to the network during training are resized to 224x224
#     horizontally flipped randomly and augmented using cutout.
#     each mini-batch contained 256 examples

from torchvision.datasets.cifar import CIFAR100 
from torchvision.transforms import Compose, RandomCrop, Pad, RandomHorizontalFlip, Resize
from torchvision.transforms import ToTensor, Normalize

from torch.utils.data import Subset

path = "/tmp/cifar100"

from PIL.Image import BICUBIC

train_transform = Compose([
    Resize(256, BICUBIC),
    RandomCrop(224),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = Compose([
    Resize(224, BICUBIC),    
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CIFAR100(root=path, train=True, transform=train_transform, download=True)
test_dataset = CIFAR100(root=path, train=False, transform=test_transform, download=False)

train_eval_indices = [random.randint(0, len(train_dataset) - 1) for i in range(len(test_dataset))]
train_eval_dataset = Subset(train_dataset, train_eval_indices)

len(train_dataset), len(test_dataset), len(train_eval_dataset)

from torch.utils.data import DataLoader

batch_size = 172
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=20, 
                          shuffle=True, drop_last=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=20, 
                         shuffle=False, drop_last=False, pin_memory=True)
eval_train_loader = DataLoader(train_eval_dataset, batch_size=batch_size, num_workers=20, 
                               shuffle=False, drop_last=False, pin_memory=True)

import torchvision.utils as vutils

# Plot some training images
batch = next(iter(train_loader))

plt.figure(figsize=(16, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow( 
    vutils.make_grid(batch[0][:16], padding=2, normalize=True).cpu().numpy().transpose((1, 2, 0))
)

batch = None
torch.cuda.empty_cache()

# Finetunning model
model.head[6].in_features, model.head[6].out_features
num_classes = 8
model.head[6] = nn.Linear(1280, num_classes)
model.head[6].in_features, model.head[6].out_features

assert torch.cuda.is_available()
assert torch.backends.cudnn.enabled, "NVIDIA/Apex:Amp requires cudnn backend to be enabled."
torch.backends.cudnn.benchmark = True

device = "cuda"
model = model.to(device)

from itertools import chain
import torch.optim as optim
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()

lr = 0.01

optimizer = optim.SGD([
    {
        "params": chain(model.stem.parameters(), model.blocks.parameters()),
        "lr": lr * 0.1,
    },
    {
        "params": model.head[:6].parameters(),
        "lr": lr * 0.2,
    },    
    {
        "params": model.head[6].parameters(), 
        "lr": lr
    }], 
    momentum=0.9, weight_decay=0.001, nesterov=True)

from torch.optim.lr_scheduler import ExponentialLR

lr_scheduler = ExponentialLR(optimizer, gamma=0.975)

try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


# Initialize Amp
model, optimizer = amp.initialize(model, optimizer, opt_level="O2", num_losses=1)

from ignite.utils import convert_tensor


def update_fn(engine, batch):
    model.train()

    x = convert_tensor(batch[0], device=device, non_blocking=True)
    y = convert_tensor(batch[1], device=device, non_blocking=True)
    
    y_pred = model(x)
    
    # Compute loss 
    loss = criterion(y_pred, y)    

    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()

    optimizer.step()
    
    return {
        "batchloss": loss.item(),
    }    