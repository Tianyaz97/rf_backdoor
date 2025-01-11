import torch
from torch import nn
from torchvision import models
from attention import *
from functools import wraps


# define the backbone model
# class FCN(nn.Module):
#     """
#     Input har data shape: (batch, n_channels=2, seq_len=256)
#     """
#     def __init__(self, n_channels=2, n_classes=16, out_channels=32, backbone=True):
#         super(FCN, self).__init__()

#         self.backbone = backbone

#         self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
#                                          nn.BatchNorm1d(32),
#                                          nn.ReLU(),
#                                          nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#                                          nn.Dropout(0.35))
#         self.conv_block2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
#                                          nn.BatchNorm1d(64),
#                                          nn.ReLU(),
#                                          nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
#         self.conv_block3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
#                                          nn.BatchNorm1d(out_channels),
#                                          nn.ReLU(),
#                                          nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

#         self.out_len = 34  # 256 -> 129 -> 65 -> 34

#         self.out_channels = out_channels
#         self.out_dim = self.out_len * self.out_channels # 32 * 34 = 1088

#         if backbone == False:
#             self.logits = nn.Linear(self.out_len * out_channels, n_classes)

#     def forward(self, x_in):
#         # x_in = x_in.permute(0, 2, 1)
#         x = self.conv_block1(x_in)
#         x = self.conv_block2(x)
#         x = self.conv_block3(x)

#         if self.backbone:
#             x = x.view(len(x), -1)
#             # x = F.normalize(x, p=2, dim=-1)
#             return None, x
#         else:
#             x_flat = x.reshape(x.shape[0], -1)
#             # x_flat = F.normalize(x_flat, p=2, dim=-1)
#             logits = self.logits(x_flat)
#             return logits, x_flat


class Identity(nn.Module):
    """
    Identity layer: gives output what it takes as input.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    

# # ResNet Model
# class Adaptive_ResNet18(nn.Module):
#     """
#     adaptive ResNet18 to fit the input shape of har data
#     """
#     def __init__(self, n_classes=6, backbone=True):
#         super(Adaptive_ResNet18, self).__init__()
#         self.backbone = backbone

#         # linear layer to expand the feature dimension to 64.
#         self.fc = nn.Linear(2, 64)

#         # pre-trained model
#         self.pretrained = models.resnet18(weights=None)
#         self.pretrained.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=2, padding=0, bias=False)
#         self.pretrained.maxpool = nn.Identity()
#         self.pretrained.fc = Identity()

#         for p in self.pretrained.parameters():
#             p.requires_grad = True
        
#         self.out_dim = 512
#         self.out_channels = 128

#         if backbone == False:
#             self.logits = nn.Linear(self.out_dim, n_classes)

#     def forward(self, x_in):
#         x_in = x_in.transpose(1, 2)
#         x = self.fc(x_in)
#         x = x.transpose(1, 2)
        
#         # add a dimension to form a 4D tensor --> (batch_size, 1, 64, 256)
#         x = x.unsqueeze(1)

#         x = self.pretrained(x)

#         if self.backbone:
#             x = x.view(len(x), -1)
#             # x = F.normalize(x, p=2, dim=-1)
#             return None, x
#         else:
#             x_flat = x.reshape(x.shape[0], -1)
#             x_flat = F.normalize(x_flat, p=2, dim=-1)
#             logits = self.logits(x_flat)
#             return logits, x_flat
        

# class Transformer(nn.Module):
#     def __init__(self, n_channels, len_sw, n_classes, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True):
#         """
#         len_sw: length of sliding window: 120 for all HAR datasets
#         """
#         super(Transformer, self).__init__()

#         self.backbone = backbone
#         self.out_dim = dim
#         self.transformer = Seq_Transformer(n_channel=n_channels, len_sw=len_sw, n_classes=n_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
#         if backbone == False:
#             self.classifier = nn.Linear(dim, n_classes)

#     def forward(self, x):
#         x = self.transformer(x)
#         if self.backbone:
#             return None, x
#         else:
#             out = self.classifier(x)
#             return out, x

        

"""
Predictor and Projector
"""
# projection head
class Projector(nn.Module):
    def __init__(self, model, bb_dim, prev_dim, dim):
        """
        bb_dim: backbone dim
        prev_dim: hidden dim
        dim: projection dim
        """
        super(Projector, self).__init__()
        if model == 'SimCLR':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim))  # 2176 -> 128
        elif model == 'byol':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim, bias=False),
                                           nn.BatchNorm1d(dim, affine=False))
        elif model == 'NNCLR':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim, bias=False),
                                           nn.BatchNorm1d(dim))
        elif model == 'TS-TCC':
            self.projector = nn.Sequential(nn.Linear(dim, bb_dim // 2),
                                           nn.BatchNorm1d(bb_dim // 2),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(bb_dim // 2, bb_dim // 4))
        else:
            raise NotImplementedError

    def forward(self, x):
        # print(x.shape) # torch.Size([256, 2176])
        x = self.projector(x)
        return x
    
    
class Predictor(nn.Module):
    def __init__(self, model, dim, pred_dim):
        super(Predictor, self).__init__()
        if model == 'SimCLR':
            pass
        elif model == 'byol':
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pred_dim, dim))
        elif model == 'NNCLR':
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pred_dim, dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.predictor(x)
        return x        


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

        
def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projector and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, DEVICE, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer
        self.DEVICE = DEVICE

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        children = [*self.net.children()]
        print('children[self.layer]:', children[self.layer])
        return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output.reshape(output.shape[0], -1)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = Projector(model='byol', bb_dim=dim, prev_dim=self.projection_hidden_size, dim=self.projection_size)
        return projector.to(hidden)

    def get_representation(self, x):

        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        if self.net.__class__.__name__ in ['AE', 'CNN_AE']:
            x_decoded, representation = self.get_representation(x)
        else:
            _, representation = self.get_representation(x)

        if len(representation.shape) == 3:
            representation = representation.reshape(representation.shape[0], -1)

        projector = self._get_projector(representation)
        projection = projector(representation)
        if self.net.__class__.__name__ in ['AE', 'CNN_AE']:
            return projection, x_decoded, representation
        else:
            return projection, representation
