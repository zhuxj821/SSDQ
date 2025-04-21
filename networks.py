import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8

class network_wrapper(nn.Module):
    def __init__(self, args):
        super(network_wrapper, self).__init__()
        self.args = args
        if args.network_audio.backbone == 'ss_dprnn':
            from models.ss_dprnn.ss_dprnn import ss_Dprnn
            self.sep_network = ss_Dprnn(args)
        else:
            raise NameError('Wrong network selection')

    def forward(self, mixture, ref):
        if self.args.network_audio.backbone =='ss_dprnn':
            if self.args.network_reference.cue == 'text':
                return self.sep_network(mixture,ref)
            else:
                raise NameError('Wrong network and reference combination selection')
        else:
            raise NameError('Wrong network selection')



