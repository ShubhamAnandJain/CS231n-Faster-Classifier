import torch.nn as nn
import torch
import torchvision.models as models
from gradinit import gradinit
from conv_norm import PreConv
from approx.src.pytorch.approx_mul_pytorch.modules.approx_Linear import approx_Linear
from approx.src.pytorch.approx_mul_pytorch.modules.approx_Conv2d import approx_Conv2d

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def SmallCNN():

  model = nn.Sequential(
    # conv_1
    nn.Conv2d(3, 32, (3, 3), padding="same"),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 32, (3, 3), padding="same"),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    
    #conv_2
    nn.Conv2d(32, 64, (3, 3), padding="same"),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 64, (3, 3), padding="same"),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    
    #fully_connected
    Flatten(),
    nn.Linear(4096, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
    nn.Softmax(dim=1)
  )

  return model

def replace_layers(model, old, new, device, bn_pres=False, mult_val=1.0):
  pres = False

  for n, module in model.named_children():
    if len(list(module.children())) > 0:
      ## compound module, go inside it
      _, npres = replace_layers(module, old, new, device, bn_pres)
      pres |= npres

    if type(module) is old:
      ## simple module
      if new == PreConv:
        pc_layer = PreConv(
          module.in_channels,
          module.out_channels,
          kernel_size=module.kernel_size,
          bias=bn_pres,
          padding=module.padding,
          stride=module.stride,
          affine=bn_pres,
          bn=False
        ).to(device)
        setattr(model, n, pc_layer)
      
      elif new == approx_Linear:
        set_bias = False
        if module.bias is not None:
          set_bias = True
        lin_layer = approx_Linear(
          module.in_features,
          module.out_features,
          bias=set_bias,
          sample_ratio=mult_val
        ).to(device)
        setattr(model, n, lin_layer)
      
      elif new == approx_Conv2d:
        conv_layer = approx_Conv2d(
          module.in_channels,
          module.out_channels,
          kernel_size=module.kernel_size,
          padding=module.padding,
          stride=module.stride,
          sample_ratio=mult_val
        ).to(device)
        setattr(model, n, conv_layer)

      elif new is None:
        setattr(model, n, nn.Identity().to(device))

      if old == nn.BatchNorm2d:
        pres = True

  return (model, pres)

def get_num_layers(model, layer):

  num_layers = 0

  for n, module in model.named_children():
    if len(list(module.children())) > 0:
      num_child = get_num_layers(module, layer)
      num_layers += num_child

    if type(module) is layer:
      num_layers += 1

  return num_layers

POSSIBLE_MODES = ["first", "last", "firstlast", "first", "last"]
# mode = (("first", 0.0), ("last", 0.0), ("firstlast", 0.0), ("first", 0.3), ("last", 0.5))


def replace_layer_from_list(model, old, new, replace_list, device, num_layers, mult_val=1.0):

  for n, module in model.named_children():

    if type(module) is old:
      if num_layers in replace_list:
        if new == approx_Linear:
          set_bias = False
          if module.bias is not None:
            set_bias = True
          lin_layer = approx_Linear(
            module.in_features,
            module.out_features,
            bias=set_bias,
            sample_ratio=mult_val
          ).to(device)
          setattr(model, n, lin_layer)
        
        elif new == approx_Conv2d:
          conv_layer = approx_Conv2d(
            module.in_channels,
            module.out_channels,
            kernel_size=module.kernel_size,
            padding=module.padding,
            stride=module.stride,
            sample_ratio=mult_val
          ).to(device)
          setattr(model, n, conv_layer)

      num_layers += 1

    if len(list(module.children())) > 0:
      num_layers = replace_layer_from_list(module, old, new, replace_list, device, num_layers, mult_val)


  return num_layers


def replace(model, layer, new_layer, mode, device, mult_val=1.0):
  num_layers= get_num_layers(model, layer)
  all_layers = list(range(num_layers))
  replace_layers_list = []

  for mode_constraint in mode:
    replace_val = mode_constraint[1]

    if(mode_constraint[0]=="first_frac"):
      replace_layers_list += all_layers[:int(num_layers * replace_val)]

    if(mode_constraint[0]=="last_frac"):
      replace_layers_list += all_layers[-int(num_layers * replace_val):]

    if(mode_constraint[0]=="first_num"):
      replace_layers_list += all_layers[:replace_val]
    
    if(mode_constraint[0]=="last_only"):
      replace_layers_list += all_layers[-replace_val:]

  replace_layers_list = list(set(replace_layers_list))
  replace_layers_list.sort()

  print(f'Replace layers list is: {replace_layers_list}')
    
  replace_layer_from_list(model, layer, new_layer, replace_layers_list, device, 0, mult_val)

  num_layers = get_num_layers(model, layer)
  new_layers = get_num_layers(model, new_layer)
  print(f'Number of og layers are {num_layers}, number of new layers are {new_layers}')

  return model


def add_convnorm(model, device):

  model, bn_pres = replace_layers(model, nn.BatchNorm2d, None, device)
  model, _ = replace_layers(model, nn.Conv2d, PreConv, device, bn_pres=bn_pres)

  return model

def approx_mult(model, device, approx_mult_params):

  mult_val = approx_mult_params['mult_val']
  mode = approx_mult_params['mode']

  model = replace(model, nn.Linear, approx_Linear, mode, device, mult_val=mult_val)
  model = replace(model, nn.Conv2d, approx_Conv2d, mode, device, mult_val=mult_val)

  return model

class ChannelWrapper(nn.Module):
  def __init__(self, model, inp_channels=1, model_channels=3):
    super().__init__()
    self.conv = nn.Conv2d(inp_channels, model_channels, 1)
    self.model = model

  def forward(self, x):

    out = self.conv(x)
    out = self.model(out)

    return out

  def string(self):

    return self.model.string()


def get_model(model_name, model_params, learning_rate, loader_train, num_channels, device):

  model_dict = {'VGG16' : models.vgg16, 'Resnet18' : models.resnet18, 'SmallCNN' : SmallCNN}
  model = model_dict[model_name]().to(device)

  if model_params.get('convnorm') is not None:
    model = add_convnorm(model, device)

  if model_params.get('approx_mult') is not None:
    model = approx_mult(model, device, model_params['approx_mult'])

  if num_channels != 3:
    model = ChannelWrapper(model, num_channels, 3).to(device)

  if model_params.get('gradinit') is not None:
    # model_params['gradinit']['lr'] = learning_rate
    model = gradinit(model, model_params['gradinit'], loader_train)

  # NOTE: assuming our model's input channels are 3

  
  return model