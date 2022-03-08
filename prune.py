import os.path as osp
import re
from collections import OrderedDict
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, ConvTranspose2d
from models import LSVC
from DVC.subnet import Bitparm

# These grad_fn pattern are flags of specific a nn.Module
CONV = ('CudnnConvolutionTransposeBackward', 'CudnnConvolutionBackward')
FC = ('MulBackward', 'MmBackward') # multiply, dot product
BN = ('NativeLayerNormBackward')
# all modules with channel should be considered
# the modules which contains NON_PASS grad_fn need to change the parameter size
# according to channels after pruning
NON_PASS = CONV + FC


def load_checkpoint(model, filename):
    checkpoint = torch.load(filename,map_location=torch.device('cuda:0'))
    state_dict = checkpoint['state_dict']
    own_state = model.state_dict()
    for name, param in state_dict.items():
        print(name,param.size(),int(param.sum()) if 'out_mask' in name or 'in_mask' in name else None)
        own_state[name].copy_(param)

def save_checkpoint(model, filename):
    state = {'state_dict': model.state_dict()}
    torch.save(state, filename)

class FisherPruningHook():
    """Use fisher information to pruning the model, must register after
    optimizer hook.

    Args:
        pruning (bool): When True, the model in pruning process,
            when False, the model is in finetune process.
            Default: True
        delta (str): "acts" or "flops", prune the model by
            "acts" or flops. Default: "acts"
        interval (int): The interval of  pruning two channels.
            Default: 10
        deploy_from (str): Path of checkpoint containing the structure
            of pruning model. Defaults to None and only effective
            when pruning is set True.
        save_flops_thr  (list): Checkpoint would be saved when
            the flops reached specific value in the list:
            Default: [0.75, 0.5, 0.25]
        save_acts_thr (list): Checkpoint would be saved when
            the acts reached specific value in the list:
            Default: [0.75, 0.5, 0.25]
    """
    def __init__(
        self,
        pruning=True,
        delta='acts',
        interval=10,
        deploy_from=None,
        resume_from=None,
        start_from=None,
        save_flops_thr=[0.75, 0.5, 0.25],
        save_acts_thr=[0.75, 0.5, 0.25],
    ):

        assert delta in ('acts', 'flops')
        self.pruning = pruning
        self.delta = delta
        self.interval = interval
        # The key of self.flops is conv module, and value of it
        # is the summation of conv's flops in forward process
        self.flops = {}
        # The key of self.acts is conv module, and value of it
        # is number of all the out feature's activations(N*C*H*W)
        # in forward process
        self.acts = {}
        # The key of self.temp_fisher_info is conv module, and value
        # is a temporary variable used to estimate fisher.
        self.temp_fisher_info = {}

        # The key of self.batch_fishers is conv module, and value
        # is the estimation of fisher by single batch.
        self.batch_fishers = {}

        # The key of self.accum_fishers is conv module, and value
        # is the estimation of parameter's fisher by all the batch
        # during number of self.interval iterations.
        self.accum_fishers = {}
        self.channels = 0
        self.delta = delta
        self.deploy_from = deploy_from
        self.resume_from = resume_from
        self.start_from = start_from

        for i in range(len(save_acts_thr) - 1):
            assert save_acts_thr[i] > save_acts_thr[i + 1]
        for i in range(len(save_flops_thr) - 1):
            assert save_flops_thr[i] > save_flops_thr[i + 1]

        self.save_flops_thr = save_flops_thr
        self.save_acts_thr = save_acts_thr

    def after_build_model(self, model):
        """Remove all pruned channels in finetune stage.

        We add this function to ensure that this happens before DDP's
        optimizer's initialization
        """

        if not self.pruning:
            for n, m in model.named_modules():
                if n: m.name = n
                add_pruning_attrs(m, pruning=self.pruning)
            load_checkpoint(model, self.deploy_from)
            deploy_pruning(model)
            
        if self.start_from is not None:
            load_checkpoint(model, self.start_from)

    def before_run(self, model):
        """Initialize the relevant variables(fisher, flops and acts) for
        calculating the importance of the channel, and use the layer-grouping
        algorithm to make the coupled module shared the mask of input
        channel."""

        self.conv_names = OrderedDict() # prunable
        self.ln_names = OrderedDict()
        self.name2module = OrderedDict()

        for n, m in model.named_modules():
            if n: m.name = n
            if self.pruning:
                add_pruning_attrs(m, pruning=self.pruning)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear) or isinstance(m, Bitparm):
                self.conv_names[m] = n
                self.name2module[n] = m
            elif isinstance(m, nn.LayerNorm):
                self.ln_names[m] = n
                self.name2module[n] = m

        if self.pruning:
            # divide the conv to several group and all convs in same
            # group used same input at least once in model's
            # forward process.
            model.eval()
            self.set_group_masks(model)
            model.train()
            # outchannel is correlated with inchannel
            self.construct_outchannel_masks()
            for conv, name in self.conv_names.items():
                self.temp_fisher_info[conv] = conv.in_mask.data.new_zeros(len(conv.in_mask)) # fisher info of every in_channel
                self.accum_fishers[conv] = conv.in_mask.data.new_zeros(len(conv.in_mask))
            for group_id in self.groups:
                module = self.groups[group_id][0]
                self.temp_fisher_info[group_id] = module.in_mask.data.new_zeros(len(module.in_mask))
                self.accum_fishers[group_id] = module.in_mask.data.new_zeros(len(module.in_mask))
            self.init_flops_acts()
            self.init_temp_fishers()
            if self.resume_from is not None:
                load_checkpoint(model, self.resume_from)
            # register forward hook
            for module, name in self.conv_names.items():
                module.register_forward_hook(self.save_input_forward_hook)

        self.print_model(model, print_flops_acts=False)

    def after_train_iter(self, itr, model):
        if not self.pruning:
            return
        # compute fisher
        for module, name in self.conv_names.items():
            self.compute_fisher_backward_hook(module)
        self.group_fishers()
        self.accumulate_fishers()
        self.init_temp_fishers()
        if itr % self.interval == 0:
            self.channel_prune()
            self.init_accum_fishers()
            self.total_flops, self.total_acts = self.update_flop_act(model)
        self.init_flops_acts()

    def update_flop_act(self, model, work_dir='work_dir/'):
        flops, acts = self.compute_flops_acts()
        if len(self.save_flops_thr):
            flops_thr = self.save_flops_thr[0]
            if flops < flops_thr:
                self.save_flops_thr.pop(0)
                path = osp.join(
                    work_dir, 'flops_{:.0f}_acts_{:.0f}.pth'.format(
                        flops * 100, acts * 100))
                save_checkpoint(model, filename=path)
        if len(self.save_acts_thr):
            acts_thr = self.save_acts_thr[0]
            if acts < acts_thr:
                self.save_acts_thr.pop(0)
                path = osp.join(
                    work_dir, 'acts_{:.0f}_flops_{:.0f}.pth'.format(
                        acts * 100, flops * 100))
                save_checkpoint(model, filename=path)
        return flops, acts

    def print_model(self, model, work_dir='work_dir/', print_flops_acts=True, print_channel=True):
        """Print the related information of the current model.

        Args:
            runner (Runner): Runner in mmcv
            print_flops_acts (bool): Print the remained percentage of
                flops and acts
            print_channel (bool): Print information about
                the number of reserved channels.
        """

        if print_flops_acts:
            flops, acts = self.update_flop_act(model, work_dir)
            print('Flops: {:.2f}%, Acts: {:.2f}%'.format(flops * 100, acts * 100))
        if print_channel:
            for module, name in self.conv_names.items():
                chans_i = int(module.in_mask.sum().cpu().numpy())
                chans_o = int(module.out_mask.sum().cpu().numpy())
                print('{}: input_channels: {}/{}, out_channels: {}/{}'.format(
                        name, chans_i, len(module.in_mask), chans_o, len(module.out_mask)))

    def compute_flops_acts(self):
        """Computing the flops and activation remains."""
        flops = 0
        max_flops = 0
        acts = 0
        max_acts = 0
        for module, name in self.conv_names.items():
            max_flop = self.flops[module]
            i_mask = module.in_mask
            o_mask = module.out_mask
            flops += max_flop / (i_mask.numel() * o_mask.numel()) * (
                i_mask.sum() * o_mask.sum())
            max_flops += max_flop
            max_act = self.acts[module]
            acts += max_act / o_mask.numel() * o_mask.sum()
            max_acts += max_act
        return flops.cpu().numpy() / max_flops, acts.cpu().numpy() / max_acts

    def init_accum_fishers(self):
        """Clear accumulated fisher info."""
        for module, name in self.conv_names.items():
            self.accum_fishers[module].zero_()
        for group in self.groups:
            self.accum_fishers[group].zero_()

    def find_pruning_channel(self, module, fisher, in_mask, info):
        """Find the the channel of a model to pruning.

        Args:
            module (nn.Conv | int ): Conv module of model or idx of self.group
            fisher(Tensor): the fisher information of module's in_mask
            in_mask (Tensor): the squeeze in_mask of modules
            info (dict): store the channel of which module need to pruning
                module: the module has channel need to pruning
                channel: the index of channel need to pruning
                min : the value of fisher / delta

        Returns:
            dict: store the current least important channel
                module: the module has channel need to be pruned
                channel: the index of channel need be to pruned
                min : the value of fisher / delta
        """
        module_info = {}
        # if hasattr(module,'name'):
        #     print(module.name, fisher.min())
        # else:
        #     print(module,fisher.min())
        if fisher.sum() > 0 and in_mask.sum() > 0:
            nonzero = in_mask.nonzero().view(-1)
            fisher = fisher[nonzero]
            min_value, argmin = fisher.min(dim=0)
            if min_value < info['min']:
                module_info['module'] = module
                module_info['channel'] = nonzero[argmin]
                module_info['min'] = min_value
        return module_info

    def single_prune(self, info, exclude=None):
        """Find the channel with smallest fisher / delta in modules not in
        group.

        Args:
            info (dict): Store the channel of which module need
                to pruning
                module: the module has channel need to pruning
                channel: the index of channel need to pruning
                min : the value of fisher / delta
            exclude (list): List contains all modules in group.
                Default: None

        Returns:
            dict: store the channel of which module need to be pruned
                module: the module has channel need to be pruned
                channel: the index of channel need be to pruned
                min : the value of fisher / delta
        """

        for module, name in self.conv_names.items():
            if exclude is not None and module in exclude:
                continue
            fisher = self.accum_fishers[module]
            in_mask = module.in_mask.view(-1)
            ancestors = self.conv2ancest[module]
            if self.delta == 'flops':
                # delta_flops is a value indicate how much flops is
                # reduced in entire forward process after we set a
                # zero in `in_mask` of a specific conv_module.
                # this affects both current and ancestor module
                # flops per channel
                in_rep = module.in_rep if type(module).__name__ == 'Linear' else 1
                delta_flops = self.flops[module] * module.out_mask.sum() / (
                    module.in_channels * module.out_channels) * in_rep
                for ancestor in ancestors:
                    out_rep = ancestor.out_rep if type(module).__name__ == 'Linear' else 1
                    delta_flops += self.flops[ancestor] * ancestor.in_mask.sum(
                    ) / (ancestor.in_channels * ancestor.out_channels) * out_rep
                fisher /= (float(delta_flops) / 1e9)
            if self.delta == 'acts':
                # activation only counts ancestors
                delta_acts = 0
                for ancestor in ancestors:
                    out_rep = ancestor.out_rep if type(module).__name__ == 'Linear' else 1
                    delta_acts += self.acts[ancestor] / ancestor.out_channels * out_rep
                fisher /= (float(max(delta_acts, 1.)) / 1e6)
            info.update(
                self.find_pruning_channel(module, fisher, in_mask, info))
        return info

    def channel_prune(self):
        """Select the channel in model with smallest fisher / delta set
        corresponding in_mask 0."""

        info = {'module': None, 'channel': None, 'min': 1e9}
        info.update(self.single_prune(info, self.group_modules))
        for group in self.groups:
            # they share the same in mask
            in_mask = self.groups[group][0].in_mask.view(-1)
            fisher = self.accum_fishers[group].double()
            if self.delta == 'flops':
                fisher /= float(self.flops[group] / 1e9)
            elif self.delta == 'acts':
                fisher /= float(self.acts[group] / 1e6)
            info.update(self.find_pruning_channel(group, fisher, in_mask, info))
        module, channel = info['module'], info['channel']
        # only modify in_mask is sufficient
        if isinstance(module, int):
            # the case for multiple modules in a group
            for m in self.groups[module]:
                m.in_mask[channel] = 0
        elif module is not None:
            # the case for single module
            module.in_mask[channel] = 0

    def accumulate_fishers(self):
        """Accumulate all the fisher during self.interval iterations."""

        for module, name in self.conv_names.items():
            self.accum_fishers[module] += self.batch_fishers[module]
        for group in self.groups:
            self.accum_fishers[group] += self.batch_fishers[group]

    def group_fishers(self):
        """Accumulate all module.in_mask's fisher and flops in same group."""
        # the case for groups
        for group in self.groups:
            self.flops[group] = 0
            self.acts[group] = 0
            # impact on group members
            for module in self.groups[group]:
                # accumulate fisher per channel per batch
                module_fisher = self.temp_fisher_info[module]
                self.temp_fisher_info[group] += module_fisher 
                # accumulate flops per in_channel per batch for each group
                if type(module).__name__ != 'Bitparm': 
                    delta_flops = self.flops[module] // module.in_channels // \
                        module.out_channels * module.out_mask.sum()
                else:
                    delta_flops = self.flops[module] // module.in_channels
                self.flops[group] += delta_flops

            # sum along the dim of batch
            self.batch_fishers[group] = self.temp_fisher_info[group]**2

            # impact on group ancestors, whose out channels are coupled with its
            # in_channels
            for module in self.ancest[group]:
                if type(module).__name__ != 'Bitparm': 
                    delta_flops = self.flops[module] // module.out_channels // \
                            module.in_channels * module.in_mask.sum()
                else:
                    delta_flops = self.flops[module] // module.out_channels
                self.flops[group] += delta_flops
                acts = self.acts[module] // module.out_channels
                self.acts[group] += acts
        # the case for single modules
        for module, name in self.conv_names.items():
            self.batch_fishers[module] = self.temp_fisher_info[module]**2

    def init_flops_acts(self):
        """Clear the flops and acts of model in last iter."""
        for module, name in self.conv_names.items():
            self.flops[module] = 0
            self.acts[module] = 0

    def init_temp_fishers(self):
        """Clear fisher info of single conv and group."""
        for module, name in self.conv_names.items():
            self.temp_fisher_info[module].zero_()
        for group in self.groups:
            self.temp_fisher_info[group].zero_()

    def save_input_forward_hook(self, module, inputs, outputs):
        """Save the input and flops and acts for computing fisher and flops or
        acts. Total flops

        Args:
            module (nn.Module): the module of register hook
        """
        layer_name = type(module).__name__
        if layer_name in ['Conv2d', 'ConvTranspose2d']:
            n, oc, oh, ow = module.output_size
            ic = module.in_channels
            kh, kw = module.kernel_size
            self.flops[module] += np.prod([n, oc, oh, ow, ic, kh, kw])
            self.acts[module] += np.prod([n, oc, oh, ow])
        elif layer_name in ['Linear']:
            n, sl, oc = module.output_size
            ic = module.weight.size(1)
            self.flops[module] += np.prod([n, sl, oc, ic])
            self.acts[module] += np.prod([n, sl, oc])
        elif layer_name in ['Bitparm']:
            n, oc, oh, ow = module.output_size
            self.flops[module] += np.prod([n, oc, oh, ow])
            self.acts[module] += np.prod([n, oc, oh, ow])
        else:
            print('Unrecognized in save_input_forward_hook:',layer_name)
            exit(0)

    def compute_fisher_backward_hook(self, module):
        # there are some bugs in torch, not using the backward hook
        """
        Args:
            module (nn.Module): module register hooks
            grad_input (tuple): tuple contains grad of input and parameters,
                grad_input[0]is the grad of input in Pytorch 1.3, it seems
                has changed in Higher version
        """
        def compute_fisher(weight, grad_weight, layer_name):
            # information per mask channel per module
            grads = weight*grad_weight
            if layer_name in ['Conv2d', 'Bitparm']:
                grads = grads.sum(-1).sum(-1).sum(0)
            elif layer_name in ['ConvTranspose2d']:
                grads = grads.sum(-1).sum(-1).sum(-1)
            elif layer_name in ['Linear']:
                grads = grads.sum(0)
                grads = grads.view(module.in_rep,-1).sum(0)
            else:
                print('Unrecognized in compute_fisher:',layer_name)
                exit(0)
            return grads

        layer_name = type(module).__name__
        weight = module.weight if layer_name not in ['Bitparm'] else module.h
        self.temp_fisher_info[module] += compute_fisher(weight, weight.grad, layer_name)

    def construct_outchannel_masks(self):
        """Register the `input_mask` of one conv to it's nearest ancestor conv,
        and name it as `out_mask`, which means the actually number of output
        feature map after pruning."""

        for conv, name in self.conv_names.items():
            for m, ancest in self.conv2ancest.items():
                if conv in ancest:
                    conv.out_mask = m.in_mask
                    break

        # make sure norm and conv output are the same    
        for bn, name in self.ln_names.items():
            conv_module = self.ln2ancest[bn][0]
            bn.out_mask = conv_module.out_mask

    def make_groups(self):
        """The modules (convolutions and BNs) connected to the same conv need
        to change the channels simultaneously when pruning.

        This function divides all modules into different groups according to
        the connections.
        """

        idx = -1
        groups, groups_ancest = {}, {}
        for module, name in reversed(self.conv_names.items()):
            added = False
            for group in groups:
                module_ancest = set(self.conv2ancest[module])
                group_ancest = set(groups_ancest[group])
                if len(module_ancest.intersection(group_ancest)) > 0:
                    groups[group].append(module)
                    groups_ancest[group] = list(
                        module_ancest.union(group_ancest))
                    added = True
                    break
            if not added:
                idx += 1
                groups[idx] = [module]
                groups_ancest[idx] = self.conv2ancest[module]
        # key is the ids the group, and value contains all conv
        # of this group
        self.groups = {}
        # key is the ids the group, and value contains all nearest
        # ancestor of this group
        self.ancest = {}
        idx = 0
        # filter the group with only one conv
        for group in groups:
            modules = groups[group]
            if len(modules) > 1:
                self.groups[idx] = modules
                self.ancest[idx] = groups_ancest[group]
                idx += 1
        # the conv's name in same group, just for debug
        # TODO remove this
        self.conv_names_group = [[item.name for item in v]
                                 for idx, v in self.groups.items()]
        # for g in self.conv_names_group:
        #     print(g)
        # exit(0)

    def set_group_masks(self, model):
        """the modules(convolutions and BN) connect to same convolutions need
        change the out channels at same time when pruning, divide the modules
        into different groups according to the connection.

        Args:
            model(nn.Module): the model contains all modules
        """
        # split into prunable or not
        # prunable group modules with coupled inchannel, update outchannel based on correlation
        # self.conv2ancest is a dict, key is all conv/deconv/linear/bitparm/layernorm instance in
        # model, value is a list which contains all [nearest] ancestor
        # bitparm and layer norm only need the mask of conv/deconv/linear to decide mask
        self.find_module_ancestors(model)
        self.make_groups()

        # list contains all the convs which are contained in a
        # group (if more the one conv has same ancestor,
        # they will be in same group)
        self.group_modules = []
        for group in self.groups:
            self.group_modules.extend(self.groups[group])

    def find_module_ancestors(self, model):
        """find the nearest module
        Args:
            loss(Tensor): the output of the network
            pattern(Tuple[str]): the pattern name

        Returns:
            dict: the key is the module match the pattern(Conv or Fc),
             and value is the list of it's nearest ancestor 
        """
        def name2ancest(n):
            if 'mvEncoder' in n:
                net_name = 'mvEncoder' 
                d = 8
            elif 'resEncoder' in n:
                net_name = 'resEncoder'
                d = 4
            elif 'respriorEncoder' in n:
                net_name = 'respriorEncoder'
                d = 3
            else:
                print('Unrecognized net name:',n)
                exit(0)
            if 'to_qkv' in n:
                a,b = re.findall(r'\d+',n)
                if a == '0' and b == '0':
                    ancest_name = [f'{net_name}.conv{d}']
                elif b == '0':
                    ancest_name = [f'{net_name}.conv{d}', f'{net_name}.layers.{int(a)-1}.2.fn.net.3']
                else:
                    ancest_name = [f'{net_name}.conv{d}', f'{net_name}.layers.{a}.0.fn.to_out.0']
            elif 'to_out' in n:
                a,b,c = re.findall(r'\d+',n)
                ancest_name = [f'{net_name}.layers.{a}.{b}.fn.to_qkv']
            elif 'norm' in n:
                a,b = re.findall(r'\d+',n)
                if b != '2':
                    ancest_name = [f'{net_name}.layers.{a}.{b}.fn.to_out.0']
                else:
                    ancest_name = [f'{net_name}.layers.{a}.{b}.fn.net.3']
            elif 'net' in n:
                a,b,c = re.findall(r'\d+',n)
                if c == '0':
                    ancest_name = [f'{net_name}.conv{d}', f'{net_name}.layers.{a}.{int(b)-1}.fn.to_out.0']
                else:
                    ancest_name = [f'{net_name}.layers.{a}.{b}.fn.net.0']
            else:
                print('Unexpected in find_module_ancestors:',n)
                exit(0)
            return ancest_name
        import re
        conv2ancest = {}
        ln2ancest = {}
        for n, m in model.named_modules():
            if type(m).__name__ not in ['Conv2d','ConvTranspose2d','Linear','LayerNorm','Bitparm']:
                continue
            # independent nets
            if 'opticFlow' in n:
                mi,ci = [int(c) for c in n if c.isdigit()]
                if ci==1:
                    ancest_name = []
                else:
                    ancest_name = [f'opticFlow.moduleBasic.{mi}.conv{ci-1}']
            elif 'mvEncoder' in n:
                if 'conv' in n:
                    a, = re.findall(r'\d+',n)
                    if a == '1':
                        ancest_name = []
                    else:
                        ancest_name = [f'mvEncoder.conv{int(a)-1}']
                elif 'layers' in n:
                    ancest_name = name2ancest(n)
                else:
                    print('Unexpected layer in mvEncoder')
            elif 'bitEstimator_mv' in n:
                a, = re.findall(r'\d+',n)
                if a == '1':
                    ancest_name = [f'mvEncoder.layers.11.2.fn.net.3']
                else:
                    ancest_name = [f'bitEstimator_mv.f{int(a)-1}']
            elif 'mvDecoder' in n:
                a, = re.findall(r'\d+',n)
                if a == '1':
                    ancest_name = [f'bitEstimator_mv.f4']
                else:
                    ancest_name = [f'mvDecoder.deconv{int(a)-1}']
            elif 'warpnet' in n:
                if 'feature_ext' in n:
                    ancest_name = []
                else:
                    cl = [int(c) for c in n if c.isdigit()]
                    if len(cl)==1:
                        ancest_name = ['warpnet.conv5.conv2']
                    elif cl[0]==0 and cl[1]==1:
                        ancest_name = ['warpnet.feature_ext']
                    else:
                        if cl[1]==1:
                            ancest_name = [f'warpnet.conv{cl[0]-1}.conv2']
                        else:
                            ancest_name = [f'warpnet.conv{cl[0]}.conv1']
            elif 'resEncoder' in n:
                if 'conv' in n:
                    a, = re.findall(r'\d+',n)
                    if a == '1':
                        ancest_name = []
                    else:
                        ancest_name = [f'resEncoder.conv{int(a)-1}']
                elif 'layers' in n:
                    ancest_name = name2ancest(n)
                else:
                    print('Unexpected layer in resEncoder')
            elif 'resDecoder' in n:
                a, = re.findall(r'\d+',n)
                if a == '1':
                    ancest_name = [f'respriorDecoder.deconv3']
                else:
                    ancest_name = [f'resDecoder.deconv{int(a)-1}']
            elif 'respriorEncoder' in n:
                if 'conv' in n:
                    a, = re.findall(r'\d+',n)
                    if a == '1':
                        ancest_name = [f'resEncoder.layers.11.2.fn.net.3']
                    else:
                        ancest_name = [f'respriorEncoder.conv{int(a)-1}']
                elif 'layers' in n:
                    ancest_name = name2ancest(n)
                else:
                    print('Unexpected layer in resEncoder')
            elif 'respriorDecoder' in n:
                a, = re.findall(r'\d+',n)
                if a == '1':
                    ancest_name = [f'bitEstimator_z.f4']
                else:
                    ancest_name = [f'respriorDecoder.deconv{int(a)-1}']
            elif 'bitEstimator_z' in n:
                a, = re.findall(r'\d+',n)
                if a == '1':
                    ancest_name = [f'respriorEncoder.layers.11.2.fn.net.3']
                else:
                    ancest_name = [f'bitEstimator_z.f{int(a)-1}']

            if type(m).__name__ in ['Conv2d','ConvTranspose2d','Linear','Bitparm']:
                conv2ancest[m] = []
            else:
                ln2ancest[m] = []
            for name in ancest_name:
                if type(m).__name__ in ['Conv2d','ConvTranspose2d','Linear','Bitparm']:
                    conv2ancest[m] += [self.name2module[name]]
                else:
                    ln2ancest[m] += [self.name2module[name]]
        self.conv2ancest = conv2ancest
        self.ln2ancest = ln2ancest


def add_pruning_attrs(module, pruning=False):
    """When module is conv, add `finetune` attribute, register `mask` buffer
    and change the origin `forward` function. When module is BN, add `out_mask`
    attribute to module.

    Args:
        conv (nn.Conv2d):  The instance of `torch.nn.Conv2d`
        pruning (bool): Indicating the state of model which
            will make conv's forward behave differently.
    """
    # TODO: mask  change to bool
    if type(module).__name__ == 'Conv2d':
        module.register_buffer(
            'in_mask', module.weight.new_ones((module.in_channels,), ))
        module.register_buffer(
            'out_mask', module.weight.new_ones((module.out_channels,), ))
        module.finetune = not pruning
        def modified_forward(m, x):
            if not m.finetune:
                mask = m.in_mask.view(1,-1,1,1)
                x = x * mask
            else:
                # if it has no ancestor
                # we need to mask it
                if x.size(1) == len(m.in_mask):
                    x = x[:,m.in_mask.bool(),:,:]
            output = F.conv2d(x, m.weight, m.bias, m.stride,
                            m.padding, m.dilation, m.groups)
            m.output_size = output.size()
            return output
        module.forward = MethodType(modified_forward, module)
    if type(module).__name__ == 'ConvTranspose2d':
        module.register_buffer(
            'in_mask', module.weight.new_ones((module.in_channels,), ))
        module.register_buffer(
            'out_mask', module.weight.new_ones((module.out_channels,), ))
        module.finetune = not pruning
        def modified_forward(m, x):
            if not m.finetune:
                mask = m.in_mask.view(1,-1,1,1)
                x = x * mask
            output = F.conv_transpose2d(x, m.weight, bias=m.bias, stride=m.stride,
                    padding=m.padding, output_padding=m.output_padding, groups=m.groups, dilation=m.dilation)
            m.output_size = output.size()
            return output
        module.forward = MethodType(modified_forward, module)
    if  type(module).__name__ == 'Linear':
        module.in_channels,module.out_channels = module.weight.size(1),module.weight.size(0)
        if 'fn.to_qkv' in module.name:
            # qkv share the same mask
            # 8 heads share the same mask
            module.out_rep = 3*8
            module.in_rep = 1
        elif 'fn.to_out.0' in module.name:
            # 8 heads share the same mask
            module.out_rep = 1
            module.in_rep = 8
        elif 'fn.net.0' in module.name:
            # gate and variable share the same mask in GEGLU
            module.out_rep = 2
            module.in_rep = 1
        else:
            module.out_rep = module.in_rep = 1
        module.register_buffer(
            'in_mask', module.weight.new_ones((module.in_channels//module.in_rep,), ))
        module.register_buffer(
            'out_mask', module.weight.new_ones((module.out_channels//module.out_rep,), ))
        module.finetune = not pruning
        def modified_forward(m, x):
            if not m.finetune:
                mask = m.in_mask.repeat(m.in_rep).view(1,1,-1)
                x = x * mask
            print(m.name,x.size(),m.weight.size(),m.bias.size())
            output = F.linear(x, m.weight, bias=m.bias)
            m.output_size = output.size()
            return output
        module.forward = MethodType(modified_forward, module)    
    if  type(module).__name__ == 'LayerNorm':
        # no need to modify layernorm during pruning since it is not computed over channels
        module.register_buffer(
            'out_mask', module.weight.new_ones((len(module.weight),), ))
    if  type(module).__name__ == 'Bitparm':
        module.register_buffer(
            'in_mask', module.h.new_ones((module.h.size(1),), ))
        module.register_buffer(
            'out_mask', module.h.new_ones((module.h.size(1),), ))
        module.finetune = not pruning
        module.in_channels = module.out_channels = module.h.size(1)
        def modified_forward(m, x):
            if not m.finetune:
                mask = m.in_mask.view(1,-1,1,1)
                x = x * mask
            if m.final:
                output = F.sigmoid(x * F.softplus(m.h) + m.b)
            else:
                x = x * F.softplus(m.h) + m.b
                output = x + F.tanh(x) * F.tanh(m.a)
            m.output_size = output.size()
            return output
        module.forward = MethodType(modified_forward, module)


def deploy_pruning(model):
    """To speed up the finetune process, We change the shape of parameter
    according to the `in_mask` and `out_mask` in it."""

    for name, module in model.named_modules():
        if type(module).__name__ == 'Conv2d':
            module.finetune = True
            requires_grad = module.weight.requires_grad
            out_mask = module.out_mask.bool()
            in_mask = module.in_mask.bool()
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = nn.Parameter(module.bias.data[out_mask])
                module.bias.requires_grad = requires_grad
            temp_weight = module.weight.data[out_mask]
            module.weight = nn.Parameter(temp_weight[:, in_mask].data)
            module.weight.requires_grad = requires_grad

        elif type(module).__name__ == 'ConvTranspose2d':
            module.finetune = True
            requires_grad = module.weight.requires_grad
            out_mask = module.out_mask.bool()
            in_mask = module.in_mask.bool()
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = nn.Parameter(module.bias.data[out_mask])
                module.bias.requires_grad = requires_grad
            temp_weight = module.weight.data[in_mask]
            module.weight = nn.Parameter(temp_weight[:, out_mask].data)
            module.weight.requires_grad = requires_grad

        elif type(module).__name__ == 'Linear':
            requires_grad = module.weight.requires_grad
            out_mask = module.out_mask.bool().repeat(module.out_rep)
            in_mask = module.in_mask.bool().repeat(module.in_rep)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = nn.Parameter(module.bias.data[out_mask])
                module.bias.requires_grad = requires_grad
            temp_weight = module.weight.data[out_mask]
            module.temp_weight = nn.Parameter(temp_weight[:, in_mask].data)
            module.weight.requires_grad = requires_grad

        elif type(module).__name__ == 'Bitparm':
            in_mask = module.in_mask.bool()
            requires_grad = module.h.requires_grad
            module.h = nn.Parameter(module.h.data[:,in_mask].data)
            module.b = nn.Parameter(module.b.data[:,in_mask].data)
            module.h.requires_grad = requires_grad
            module.b.requires_grad = requires_grad

        elif type(module).__name__ == 'LayerNorm':
            out_mask = module.out_mask.bool()
            requires_grad = module.weight.requires_grad
            module.weight = nn.Parameter(module.weight.data[out_mask].data)
            module.bias = nn.Parameter(module.bias.data[out_mask].data)
            module.weight.requires_grad = requires_grad
            module.bias.requires_grad = requires_grad

# to do: make sure linear works fine

if __name__ == '__main__':
    data = torch.randn(7,3,256,256).cuda()
    model = LSVC('LSVC-A').cuda()
    hook = FisherPruningHook(deploy_from=f'backup/LSVC-A/LSVC-A-3P_best.pth')
    hook.after_build_model(model)
    hook.before_run(model)
    x_hat, x_mc, x_wp, rec_loss, warp_loss, mc_loss, bpp_res, bpp = model(data)
    loss = rec_loss*model.r + bpp

    loss.backward()
    hook.after_train_iter(0)
    print(hook.total_flops,hook.total_acts)
