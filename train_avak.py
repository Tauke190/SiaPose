import os
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms
from tqdm import tqdm
from sia import get_sia, HungarianMatcher, SetCriterion, PostProcess
from datasets import AVA, K700, textaugavak

from util.box_ops import box_cxcywh_to_xyxy
from util.misc import reduce_dict
import numpy as np

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import json

import argparse

parser = argparse.ArgumentParser(description="finetine ViCLIP for closed-set actdet")
parser.add_argument("-SIZE", metavar="SIZE", type=str, default='b16',
                    help="B16 or L14")
parser.add_argument("-FRAMES", metavar="FRAMES", type=int, default=9,
                    help="number of input frames")
parser.add_argument("-BS", metavar="BS", type=int, default=100,
                    help="batch size")
parser.add_argument("-WORKERS", metavar="WORKERS", type=int, default=8,
                    help="num of workers for dataloader")
parser.add_argument("-EPOCH", metavar="EPOCH", type=int, default=6,
                    help="num of epochs")
parser.add_argument("-LR", metavar="LR", type=float, default=0.00001,
                    help="learning rate")
parser.add_argument("-WIDTH", metavar="WIDTH", type=int, default=320,
                    help="width of video to resize to")
parser.add_argument("-HEIGHT", metavar="HEIGHT", type=int, default=240,
                    help="height of video to resize to")
parser.add_argument("-JSON", metavar="JSON", type=str, default='stats.json',
                    help="output json name")
                    
parser.add_argument("-DET", metavar="DET", type=int, default=10,
                    help="number of [det] tokens to use")
parser.add_argument("--TXTLORA", action='store_true',
                    help="use LoRA on text encoder")
    
parser.add_argument("-TRAIN", metavar="train", type=str, default='AVAK',
                    help="Train on AVA, AVA-K or K700")

parser.add_argument("-AVA", metavar="AVA", type=str,
                    help="AVA video directory")
parser.add_argument("-RATEAVA", metavar="RATEAVA", type=int, default=8,
                    help="sampling rate of input frames")
parser.add_argument("-ANNOTRAINAVA", metavar="ANNOTRAINAVA", type=str, default='anno/ava_train_v2.2.csv',
                    help="AVA train anno path")
parser.add_argument("-ANNOVALAVA", metavar="ANNOVALAVA", type=str, default='anno/ava_val_v2.2.csv',
                    help="AVA val anno path")
parser.add_argument("-ANNOLISTAVA", metavar="ANNOLISTAVA", type=str, default='anno/ava_action_list_v2.2.pbtxt',
                    help="AVA list of actions anno path")
parser.add_argument("-ANNOLISTAVA2019", metavar="ANNOLISTAVA2019", type=str, default='anno/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
                    help="AVA 60 list of actions anno path")
                   
parser.add_argument("-KINETICS", metavar="KINETICS", type=str,
                    help="KINETICS video directory")
parser.add_argument("-RATEKINETICS", metavar="RATEKINETICS", type=int, default=8,
                    help="sampling rate of input frames")
parser.add_argument("-ANNOTRAINKINETICS", metavar="ANNOTRAINKINETICS", type=str, default='anno/kinetics_train_v1.0.csv',
                    help="KINETICS train anno path")
parser.add_argument("-ANNOVALKINETICS", metavar="ANNOVALKINETICS", type=str, default='anno/kinetics_val_v1.0.csv',
                    help="KINETICS val anno path")
parser.add_argument("-ANNOLISTKINETICS", metavar="ANNOLISTAVA", type=str, default='anno/kinetics_700_labels.csv',
                    help="KINETICS list of actions anno path")
                    
parser.add_argument("--TXTAUG", action='store_true',
                    help="augment text labels")
parser.add_argument("--DISABLEK700", action='store_true',
                    help="DO not use K700 labels")
parser.add_argument("-AWS", metavar="AWS", type=str, default='',
                    help="optional aligned weak-supervision annotation json")
                    
parser.add_argument("--SAVE", action='store_true',
                    help="save model weights after test")
args = parser.parse_args()

assert args.SIZE in ('b16', 'l14'), 'choose either b16 or l14'
assert '.json' in args.JSON, 'filename must end in .json'
if args.ANNOVALAVA == '':
    args.ANNOVALAVA = None

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
def collate_fn(batch):
        batch = list(zip(*batch))
        batch[0] = torch.stack(batch[0])
        return tuple(batch)

def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)
    ##############
    # Load Model #
    ##############
    if args.SIZE == 'b16':
        pretrain = "weights/viclip/ViCLIP-B_InternVid-FLT-10M.pth"

        model = get_sia(size='b', pretrain=pretrain, det_token_num=args.DET, text_lora=args.TXTLORA, num_frames=args.FRAMES)['sia']
    else:
        pretrain = "weights/viclip/ViCLIP-L_InternVid-FLT-10M.pth"

        model = get_sia(size='l', pretrain=pretrain, det_token_num=args.DET, text_lora=args.TXTLORA, num_frames=args.FRAMES)['sia']

    model.train()
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    print('Trainable parameters:')
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)
    print()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    parameters = model.parameters()
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Total Parameters: %.3fM' % parameters)
    print()
    del parameters

    ################
    # Load Dataset #
    ################
    video_input_num_frames = args.FRAMES

    batch_size = args.BS
    num_workers = args.WORKERS
    tfs = v2.Compose([v2.Resize((args.HEIGHT, args.WIDTH)),
                      v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                      
    enable_k700 = not args.DISABLEK700
    if enable_k700:
        print('Use weakly-supervised k700 labels')
    else:
        print('Not using weakly-supervised k700 labels')

    assert args.TRAIN in ['AVA', 'AVAK', 'K700'], 'train dataset must be AVA, AVAK or K700'
    if args.TRAIN == 'AVA':
        train_ava = AVA(args.AVA, args.ANNOTRAINAVA, args.ANNOVALAVA, args.ANNOLISTAVA, transforms=tfs, frames=args.FRAMES, rate=args.RATEAVA, split ='train')
        train_dataset = train_ava
    elif args.TRAIN == 'AVAK':
        train_ava = AVA(args.AVA, args.ANNOTRAINAVA, args.ANNOVALAVA, args.ANNOLISTAVA, transforms=tfs, frames=args.FRAMES, rate=args.RATEAVA, split ='train')
        train_kinetics = K700(args.KINETICS, args.ANNOTRAINKINETICS, args.ANNOVALKINETICS, clsfile=args.ANNOLISTAVA, csvfile=args.ANNOLISTKINETICS, transforms=tfs, frames=args.FRAMES, rate=args.RATEKINETICS, split='train', enable_k700_labels=enable_k700, aligned_anno=args.AWS)
        train_dataset = torch.utils.data.ConcatDataset((train_ava, train_kinetics))
    elif args.TRAIN == 'K700':
        train_kinetics = K700(args.KINETICS, args.ANNOTRAINKINETICS, args.ANNOVALKINETICS, clsfile=args.ANNOLISTAVA, csvfile=args.ANNOLISTKINETICS, transforms=tfs, frames=args.FRAMES, rate=args.RATEKINETICS, split='train', enable_k700_labels=enable_k700, aligned_anno=args.AWS)
        train_dataset = train_kinetics
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=num_workers, sampler=DistributedSampler(train_dataset))

    ########
    # Main #
    ########
    cost_class = 2 #1
    cost_bbox = 2 #5
    cost_giou = 2 #2
    cost_human = 2 #1

    matcher = HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou, cost_human=cost_human)
    weight_dict = {'loss_ce': cost_class, 'loss_bbox': cost_bbox, 'loss_giou': cost_giou, 'loss_human': cost_human}
    eos_coef = 0.1
    criterion = SetCriterion(matcher=matcher,
                             weight_dict=weight_dict,
                             eos_coef=eos_coef,
                             losses=['labels', 'boxes', 'cardinality', 'human'])
    criterion.to(rank)
    postprocessors = {'bbox': PostProcess()}

    epochs = args.EPOCH
    lr = args.LR
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    print('Experiment:', args.JSON[:-5])
    expdir = os.path.join('weights', args.JSON[:-5])
    if not os.path.exists(expdir):
        os.mkdir(expdir)

    iter = 0
    for epoch in range(epochs):
        print('EPOCH :', epoch)
        train_loader.sampler.set_epoch(epoch)
        reset = 0
        
        for samples, targets in tqdm(train_loader):
            optimizer.zero_grad()
            samples = samples.to(rank)

            #extract captions and temp label mapping
            captions = list(set(sum([ele for ele in sum([t['text_labels'] for t in targets],[])],[])))
            captionstoidx = {v: k for k, v in enumerate(captions)}
            temp_num_classes = len(captions)
            for t in targets:
                t['boxes'] = t['boxes'].to(rank)
                intlabels = []
                for ele in t['text_labels']:
                    intlabel = torch.tensor([captionstoidx[caption] for caption in ele])
                    intlabel = F.one_hot(intlabel, num_classes=temp_num_classes).sum(dim=0)
                    intlabels.append(intlabel)
                intlabels = torch.stack(intlabels).to(rank)
                t['labels'] = intlabels
                
            if args.TXTAUG:
                captions = textaugavak(captions)

            #backprop
            outputs = model(samples, captions)
            
            loss_dict = criterion(outputs, targets, temp_num_classes)
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            '''
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            '''
            losses.backward()
            optimizer.step()

            loss_value = losses.cpu().detach().item() #losses_reduced_scaled.item()
            print('batch loss :', round(loss_value,3))

            reset += 1
            if reset >= len(train_loader) // 1: # Test Eval
                if args.SAVE:
                    torch.save(model.module.state_dict(), os.path.join(expdir, 'avak_'+args.SIZE+'_'+str(iter)+'.pt'))
                model.eval()
                reset = 0
                iter += 1
                model.train()

    destroy_process_group()
    
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print('Number of GPUs:', world_size)
    mp.spawn(main, args=(world_size, ), nprocs=world_size)
