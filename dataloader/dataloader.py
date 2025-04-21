import numpy as np

def dataloader_wrapper(args, partition):
    if args.network_reference.cue == 'text':
        from .dataset_text_org import get_dataloader_text as get_dataloader
    else:
        raise NameError('Wrong reference for dataloader selection')
    return get_dataloader(args, partition)
    






















