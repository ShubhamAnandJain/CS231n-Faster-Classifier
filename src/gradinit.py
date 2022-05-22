import torch
import torch.nn as nn
import os
import math
import numpy as np
import argparse

def get_default_args():

  parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
  parser.add_argument('--resume', default='', type=str,
                      help='resume from checkpoint')
  parser.add_argument('--seed', default=0, type=int,
                      help='rng seed')
  parser.add_argument('--alpha', default=1., type=float,
                      help='interpolation strength (uniform=1., ERM=0.)')
  parser.add_argument('--wd', default=1e-4, type=float,
                      help='weight decay (default=1e-4)')
  parser.add_argument('--batchsize', default=128, type=int,
                      help='batch size per GPU (default=128)')
  parser.add_argument('--n_epoch', default=200, type=int,
                      help='total number of epochs')
  parser.add_argument('--base_lr', default=0.0003, type=float,
                      help='base learning rate (default=0.1)')
  parser.add_argument('--train-clip', default=-1, type=float,
                      help='Clip the gradient norm during training.')
  parser.add_argument('--expname', default="default", type=str)
  parser.add_argument('--no_bn', default=False, action='store_true')
  parser.add_argument('--dataset', default='cifar10', type=str)
  parser.add_argument('--cutout', default=False, action='store_true')
  parser.add_argument('--train-loss', default='ce', type=str, choices=['ce', 'mixup'])

  parser.add_argument('--metainit', default=False, action='store_true',
                      help='Whether to use MetaInit.')
  parser.add_argument('--gradinit', default=True, action='store_true',
                      help='Whether to use GradInit.')
  parser.add_argument('--gradinit-lr', default=1e-3, type=float,
                      help='The learning rate of GradInit.')
  parser.add_argument('--gradinit-iters', default=390, type=int,
                      help='Total number of iterations for GradInit.')
  parser.add_argument('--gradinit-alg', default='sgd', type=str,
                      help='The target optimization algorithm, deciding the direction of the first gradient step.')
  parser.add_argument('--gradinit-eta', default=0.1, type=float,
                      help='The eta in GradInit.')
  parser.add_argument('--gradinit-min-scale', default=0.01, type=float,
                      help='The lower bound of the scaling factors.')
  parser.add_argument('--gradinit-grad-clip', default=1, type=float,
                      help='Gradient clipping (per dimension) for GradInit.')
  parser.add_argument('--gradinit-gamma', default=float('inf'), type=float,
                      help='The gradient norm constraint.')
  parser.add_argument('--gradinit-normalize-grad', default=False, action='store_true',
                      help='Whether to normalize the gradient for the algorithm A.')
  parser.add_argument('--gradinit-resume', default='', type=str,
                      help='Path to the gradinit or metainit initializations.')
  parser.add_argument('--gradinit-bsize', default=-1, type=int,
                      help='Batch size for GradInit. Set to -1 to use the same batch size as training.')
  parser.add_argument('--batch-no-overlap', default=False, action='store_true',
                      help=r'Whether to make \tilde{S} and S different.')

  args = parser.parse_args(args=[])

  return args

class RescaleAdam(torch.optim.Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 min_scale=0, grad_clip=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, amsgrad=amsgrad, min_scale=min_scale, grad_clip=grad_clip)
        super(RescaleAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RescaleAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None, is_constraint=False):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad_list = []
        alphas = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # State initialization
                amsgrad = group['amsgrad']
                state = self.state[p]
                if len(state) == 0:
                    state['alpha'] = 1.
                    state['init_norm'] = p.norm().item()
                    state['step'] = 0
                    state['cons_step'] = 0
                    # Exponential moving average of gradient values for the weight norms
                    state['exp_avg'] = 0
                    # Exponential moving average of squared gradient values for the weight norms
                    state['exp_avg_sq'] = 0
                    state['cons_exp_avg'] = 0
                    # state['cons_exp_avg_sq'] = 0
                    # if amsgrad:
                    #     # Maintains max of all exp. moving avg. of sq. grad. values
                    #     state['max_exp_avg_sq'] = 0
                # alphas.append(state['alpha'])

                curr_norm = p.data.norm().item()
                if state['init_norm'] == 0 or curr_norm == 0:
                    # pdb.set_trace()
                    continue # typical for biases

                grad = torch.sum(p.grad * p.data).item() * state['init_norm'] / curr_norm
                # grad_list.append(grad)

                if group['grad_clip'] > 0:
                    grad = max(min(grad, group['grad_clip']), -group['grad_clip'])
                # Perform stepweight decay
                # if group['weight_decay'] > 0:
                #     p.mul_(1 - group['lr'] * group['weight_decay'])
                beta1, beta2 = group['betas']
                if is_constraint:
                    state['cons_step'] += 1
                    state['cons_exp_avg'] = state['cons_exp_avg'] * beta1 + grad * (1 - beta1)
                    # state['cons_exp_avg_sq'] = state['cons_exp_avg_sq'] * beta2 + (grad * grad) * (1 - beta2)

                    steps = state['cons_step']
                    exp_avg = state['cons_exp_avg']
                    # exp_avg_sq = state['cons_exp_avg_sq']
                else:
                    # pdb.set_trace()
                    state['step'] += 1
                    state['exp_avg'] = state['exp_avg'] * beta1 + grad * (1 - beta1)

                    steps = state['step']
                    exp_avg = state['exp_avg']

                state['exp_avg_sq'] = state['exp_avg_sq'] * beta2 + (grad * grad) * (1 - beta2)
                exp_avg_sq = state['exp_avg_sq']

                bias_correction1 = 1 - beta1 ** steps
                bias_correction2 = 1 - beta2 ** (state['cons_step'] + state['step'])

                # Decay the first and second moment running average coefficient
                # if amsgrad:
                #     # Maintains the maximum of all 2nd moment running avg. till now
                #     state['max_exp_avg_sq'] = max(state['max_exp_avg_sq'], state['exp_avg_sq'])
                #     # Use the max. for normalizing running avg. of gradient
                #     denom = math.sqrt(state['max_exp_avg_sq'] / bias_correction2) + group['eps']
                # else:
                denom = math.sqrt(exp_avg_sq / bias_correction2) + group['eps']

                step_size = group['lr'] / bias_correction1

                # update the parameter
                state['alpha'] = max(state['alpha'] - step_size * exp_avg / denom, group['min_scale'])
                p.data.mul_(state['alpha'] * state['init_norm'] / curr_norm)

        # print(alphas)
        # print(grad_list)
        # print(max(grad_list), min(grad_list), max(alphas), min(alphas))
        # pdb.set_trace()
        return loss

    def reset_momentums(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                if len(state) == 0:
                    state['alpha'] = 1.
                    state['init_norm'] = p.norm().item()
                    state['step'] = 0
                    # Exponential moving average of gradient values for the weight norms
                    state['exp_avg'] = 0
                    # Exponential moving average of squared gradient values for the weight norms
                    state['exp_avg_sq'] = 0
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = 0
                else:
                    state['step'] = 0
                    # Exponential moving average of gradient values for the weight norms
                    state['exp_avg'] = 0
                    # Exponential moving average of squared gradient values for the weight norms
                    state['exp_avg_sq'] = 0
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = 0


def set_bn_modes(net):
    """Switch the BN layers into training mode, but does not track running stats.
    """
    for n, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            m.track_running_stats = False

def recover_bn_modes(net):
    for n, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True

class Scale(torch.nn.Module):
    def __init__(self):
        super(Scale, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.weight


class Bias(torch.nn.Module):
    def __init__(self):
        super(Bias, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.bias

def get_ordered_params(net):
    param_list = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            param_list.append(m.weight)
            if m.bias is not None:
                param_list.append(m.bias)
        elif isinstance(m, Scale):
            param_list.append(m.weight)
        elif isinstance(m, Bias):
            param_list.append(m.bias)

    return param_list

def recover_params(net):
    """Reset the weights to the original values without the gradient step
    """

    def recover_param_(module, name):
        delattr(module, name)
        setattr(module, name, getattr(module, name + '_prev'))
        del module._parameters[name + '_prev']

    for n, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            recover_param_(m, 'weight')
            if m.bias is not None:
                recover_param_(m, 'bias')
        elif isinstance(m, Scale):
            recover_param_(m, 'weight')
        elif isinstance(m, Bias):
            recover_param_(m, 'bias')

def set_param(module, name, alg, eta, grad):
    weight = getattr(module, name)
    # remove this parameter from parameter list
    del module._parameters[name]

    # compute the update steps according to the optimizers
    if alg.lower() == 'sgd':
        gstep = eta * grad
    elif alg.lower() == 'adam':
        gstep = eta * grad.sign()
    else:
        raise RuntimeError("Optimization algorithm {} not defined!".format(alg))

    # add the updated parameter as the new parameter
    module.register_parameter(name + '_prev', weight)

    # recompute weight before every forward()
    updated_weight = weight - gstep.data
    setattr(module, name, updated_weight)

def get_scale_stats(model, optimizer):
    stat_dict = {}
    # all_s_list = [p.norm().item() for n, p in model.named_parameters() if 'bias' not in n]
    all_s_list = []
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            all_s_list.append(optimizer.state[p]['alpha'])
    stat_dict['s_max'] = max(all_s_list)
    stat_dict['s_min'] = min(all_s_list)
    stat_dict['s_mean'] = np.mean(all_s_list)
    all_s_list = []
    for n, p in model.named_parameters():
        if 'bias' not in n:
            all_s_list.append(optimizer.state[p]['alpha'])
    stat_dict['s_weight_max'] = max(all_s_list)
    stat_dict['s_weight_min'] = min(all_s_list)
    stat_dict['s_weight_mean'] = np.mean(all_s_list)

    return stat_dict


def get_batch(data_iter, data_loader):
    try:
        inputs, targets = next(data_iter)
    except:
        data_iter = iter(data_loader)
        inputs, targets = next(data_iter)
    inputs, targets = inputs.cuda(), targets.cuda()
    return data_iter, inputs, targets

def take_opt_step(net, grad_list, alg='sgd', eta=0.1):
    """Take the initial step of the chosen optimizer.
    """
    assert alg.lower() in ['adam', 'sgd']

    idx = 0
    for n, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            grad = grad_list[idx]
            set_param(m, 'weight', alg, eta, grad)
            idx += 1

            if m.bias is not None:
                grad = grad_list[idx]
                set_param(m, 'bias', alg, eta, grad)
                idx += 1
        elif isinstance(m, Scale):
            grad = grad_list[idx]
            set_param(m, 'weight', alg, eta, grad)
            idx += 1
        elif isinstance(m, Bias):
            grad = grad_list[idx]
            set_param(m, 'bias', alg, eta, grad)
            idx += 1

def gradinit(net, gradinit_args, dataloader):

    args = get_default_args()

    args.__dict__.update(gradinit_args)

    if args.gradinit_resume:
        print("Resuming GradInit model from {}".format(args.gradinit_resume))
        sdict = torch.load(args.gradinit_resume)
        net.load_state_dict(sdict)
        return

    # if isinstance(net, torch.nn.DataParallel):
    #     net_top = net.module
    # else:
    #     net_top = net

    bias_params = [p for n, p in net.named_parameters() if 'bias' in n]
    weight_params = [p for n, p in net.named_parameters() if 'weight' in n]

    optimizer = RescaleAdam([{'params': weight_params, 'min_scale': args.gradinit_min_scale, 'lr': args.lr},
                                      {'params': bias_params, 'min_scale': 0, 'lr': args.lr}],
                                     grad_clip=args.gradinit_grad_clip)

    criterion = nn.CrossEntropyLoss()

    net.eval() # This further shuts down dropout, if any.

    set_bn_modes(net) # Should be called after net.eval()


    total_loss, total_l0, total_l1, total_residual, total_gnorm = 0, 0, 0, 0, 0
    total_sums, total_sums_gnorm = 0, 0
    cs_count = 0
    total_iters = 0
    obj_loss, updated_loss, residual = -1, -1, -1
    data_iter = iter(dataloader)
    # get all the parameters by order
    params_list = get_ordered_params(net)
    while True:
        eta = args.gradinit_eta

        # continue
        # get the first half of the minibatch
        data_iter, init_inputs_0, init_targets_0 = get_batch(data_iter, dataloader)

        # Get the second half of the data.
        data_iter, init_inputs_1, init_targets_1 = get_batch(data_iter, dataloader)

        init_inputs = torch.cat([init_inputs_0, init_inputs_1])
        init_targets = torch.cat([init_targets_0, init_targets_1])
        # compute the gradient and take one step
        outputs = net(init_inputs)
        init_loss = criterion(outputs, init_targets)

        all_grads = torch.autograd.grad(init_loss, params_list, create_graph=True)

        # Compute the loss w.r.t. the optimizer
        if args.gradinit_alg.lower() == 'adam':
            # grad-update inner product
            gnorm = sum([g.abs().sum() for g in all_grads])
            loss_grads = all_grads
        else:
            gnorm_sq = sum([g.square().sum() for g in all_grads])
            gnorm = gnorm_sq.sqrt()
            if args.gradinit_normalize_grad:
                loss_grads = [g / gnorm for g in all_grads]
            else:
                loss_grads = all_grads

        total_gnorm += gnorm.item()
        total_sums_gnorm += 1
        if gnorm.item() > args.gradinit_gamma:
            # project back into the gradient norm constraint
            optimizer.zero_grad()
            gnorm.backward()
            optimizer.step(is_constraint=True)

            cs_count += 1
        else:
            # take one optimization step
            take_opt_step(net, loss_grads, alg=args.gradinit_alg, eta=eta)

            total_l0 += init_loss.item()

            data_iter, inputs_2, targets_2 = get_batch(data_iter, dataloader)
            if args.batch_no_overlap:
                # sample a new batch for the half
                data_iter, init_inputs_0, init_targets_0 = get_batch(data_iter, dataloader)
            updated_inputs = torch.cat([init_inputs_0, inputs_2])
            updated_targets = torch.cat([init_targets_0, targets_2])

            # compute loss using the updated network
            # net_top.opt_mode(True)
            updated_outputs = net(updated_inputs)
            # net_top.opt_mode(False)
            updated_loss = criterion(updated_outputs, updated_targets)

            # If eta is larger, we should expect obj_loss to be even smaller.
            obj_loss = updated_loss / eta

            recover_params(net)
            optimizer.zero_grad()
            obj_loss.backward()
            optimizer.step(is_constraint=False)
            total_l1 += updated_loss.item()

            total_loss += obj_loss.item()
            total_sums += 1

        total_iters += 1
        if (total_sums_gnorm > 0 and total_sums_gnorm % 10 == 0) or total_iters == args.gradinit_iters:
            stat_dict = get_scale_stats(net, optimizer)
            print_str = "GradInit: Iter {}, obj iters {}, eta {:.3e}, constraint count {} loss: {:.3e} ({:.3e}), init loss: {:.3e} ({:.3e}), update loss {:.3e} ({:.3e}), " \
                        "total gnorm: {:.3e} ({:.3e})\t".format(
                total_sums_gnorm, total_sums, eta, cs_count,
                float(obj_loss), total_loss / total_sums if total_sums > 0 else -1,
                float(init_loss), total_l0 / total_sums if total_sums > 0 else -1,
                float(updated_loss), total_l1 / total_sums if total_sums > 0 else -1,
                float(gnorm), total_gnorm / total_sums_gnorm)

            for key, val in stat_dict.items():
                print_str += "{}: {:.2e}\t".format(key, val)
            print(print_str)

        if total_iters == args.gradinit_iters:
            break

    recover_bn_modes(net)
    if not os.path.exists('chks'):
        os.makedirs('chks')
    torch.save(net.state_dict(), 'chks/{}_init.pth'.format(args.expname))

    return net