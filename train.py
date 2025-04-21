import yamlargparse, os, random
import numpy as np
import torch

from dataloader.dataloader import dataloader_wrapper
from solver import Solver


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTORCH_SEED'] = str(args.seed)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    args.device = device

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', rank=args.local_rank, init_method='env://', world_size=args.world_size)

    from networks import network_wrapper
    model = network_wrapper(args)
    if args.distributed: model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    if (args.distributed and args.local_rank ==0) or args.distributed == False: 
        print("started on " + args.checkpoint_dir + '\n')
        print(args)
        print("\nTotal number of parameters: {} \n".format(sum(p.numel() for p in model.parameters())))
        print("\nTotal number of trainable parameters: {} \n".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_learning_rate)

    train_sampler, train_generator = dataloader_wrapper(args,'train')
    _, val_generator = dataloader_wrapper(args, 'val')
    _, test_generator = dataloader_wrapper(args, 'test')
    args.train_sampler=train_sampler


    solver = Solver(args=args,
                model = model,
                optimizer = optimizer,
                train_data = train_generator,
                validation_data = val_generator,
                test_data = test_generator
                ) 
    if not args.evaluate_only:
        solver.train()

    # run evaluation script
    if (args.distributed and args.local_rank ==0) or args.distributed == False: 
        print("Start evaluation")
        args.batch_size=1
        args.max_length = 100
        args.distributed = False
        args.world_size = 1
        _, test_generator = dataloader_wrapper(args, 'test')
        solver.evaluate(test_generator)


if __name__ == '__main__':
    parser = yamlargparse.ArgumentParser("Settings")
    
    # Log and Visulization
    parser.add_argument('--seed', type=int)  
    parser.add_argument('--use_cuda', default=1, type=int, help='use cuda')

    parser.add_argument('--config', help='config file path', action=yamlargparse.ActionConfigFile) 
    parser.add_argument('--checkpoint_dir', type=str, help='the name of the log')
    parser.add_argument('--train_from_last_checkpoint', type=int, help='whether to train from a checkpoint, includes model weight, optimizer settings')
    parser.add_argument('--evaluate_only',  type=int, default=0, help='Only perform evaluation')

    # optimizer
    parser.add_argument('--loss_type', type=str, help='snr or sisnr')
    parser.add_argument('--init_learning_rate',  type=float, help='Init learning rate')
    parser.add_argument('--lr_warmup',  type=int, default=0, help='whether to perform lr warmup')
    parser.add_argument('--max_epoch', type=int, help='Number of maximum epochs')
    parser.add_argument('--clip_grad_norm',  type=float, help='Gradient norm threshold to clip')

    # dataset settings
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--accu_grad',type=int, help='whether to accumulate grad')
    parser.add_argument('--effec_batch_size',type=int, help='effective Batch size')
    parser.add_argument('--max_length', type=int, help='max_length of mixture in training')
    parser.add_argument('--num_workers', type=int, help='Number of workers to generate minibatch')
    
    # network settings
    parser.add_argument('--causal', type=int, help='whether the newtwork is causal')
    parser.add_argument('--network_reference', type=dict, help='the nature of auxilary reference signal')
    parser.add_argument('--network_audio', type=dict, help='a dictionary that contains the network parameters')
    parser.add_argument('--init_from', type=str, help='whether to initilize the model weights from a pre-trained checkpoint')

    # others
    parser.add_argument('--mix_lst_path', type=str)
    parser.add_argument('--noise_direc', type=str)
    parser.add_argument('--audio_direc', type=str)
    parser.add_argument('--reference_direc', type=str)
    parser.add_argument('--speaker_no', type=int)
    parser.add_argument('--audio_sr',  type=int, help='audio sampling_rate')
    parser.add_argument('--ref_sr',  type=int, help='reference signal sampling_rate')

    parser.add_argument('--region_type', type=str)
    parser.add_argument('--speaker_type', type=str)
    parser.add_argument('--SNR', type=int)
    # Distributed training
    parser.add_argument("--local-rank", default=0, type=int) 
    parser.add_argument("--num_channel", default=0, type=int)
    parser.add_argument("--num_layer", default=0, type=int)

    args = parser.parse_args()


    # check for single- or multi-GPU training
    args.distributed = False
    args.world_size = 1
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])
    assert torch.backends.cudnn.enabled, "cudnn needs to be enabled"

    main(args)


