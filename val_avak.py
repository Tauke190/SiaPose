import os
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms
from tqdm import tqdm
from sia import get_sia, HungarianMatcher, SetCriterion, PostProcess
from datasets import AVA, K700, UCF24, HMDB21, textaugava, textaugucf24, textaughmdb21, avatextaug, ucf24textaug, hmdb21textaug

from util.box_ops import box_cxcywh_to_xyxy
from util.misc import reduce_dict
from torchmetrics.detection import MeanAveragePrecision
import numpy as np

import json
from pprint import pprint

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
    
parser.add_argument("-VAL", metavar="val", type=str, default='AVAK',
                    help="Val on AVA, AVA-K or K700")

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
parser.add_argument("-ANNOLISTKINETICS", metavar="ANNOLISTKINETICS", type=str, default='anno/kinetics_700_labels.csv',
                    help="KINETICS list of actions anno path")
                    
parser.add_argument("-UCF24", metavar="UCF24", type=str,
                    help="UCF24 video directory")
parser.add_argument("-RATEUCF24", metavar="RATE", type=int, default=7,
                    help="sampling rate of input frames")
parser.add_argument("-ANNOUCF24", metavar="ANNOUCF24", type=str, default='anno/UCF101v2-GT.pkl',
                    help="ucf24 anno path")

parser.add_argument("-HMDB21", metavar="HMDB21", type=str,
                    help="HMDB21 video directory")                
parser.add_argument("-ANNOHMDB21", metavar="ANNOHMDB21", type=str, default='anno/JHMDB-GT.pkl',
                    help="hmdb21 anno path")
                    
parser.add_argument("--TXTAUG", action='store_true',
                    help="augment text labels")

parser.add_argument("-PRETRAINED", metavar="PRETRAINED", type=str, default='',
                    help="pretrained weights to use")

args = parser.parse_args()

assert args.SIZE in ('b16', 'l14'), 'choose either b16 or l14'
assert '.json' in args.JSON, 'filename must end in .json'
if args.ANNOVALAVA == '':
    args.ANNOVALAVA = None

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('Using device:', device)

##############
# Load Model #
##############
if args.SIZE == 'b16':
    pretrain = "weights/viclip/ViCLIP-B_InternVid-FLT-10M.pth"

    model = get_sia(size='b', pretrain=pretrain, det_token_num=args.DET, text_lora=args.TXTLORA, num_frames=args.FRAMES)['sia']
else:
    pretrain = "weights/viclip/ViCLIP-L_InternVid-FLT-10M.pth"

    model = get_sia(size='l', pretrain=pretrain, det_token_num=args.DET, text_lora=args.TXTLORA, num_frames=args.FRAMES)['sia']

model.to(device)

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

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0])
    return tuple(batch)

if args.VAL == 'AVA':
    test_ava = AVA(args.AVA, args.ANNOTRAINAVA, args.ANNOVALAVA, args.ANNOLISTAVA2019, transforms=tfs, frames=args.FRAMES, rate=args.RATEAVA, split ='val')
    test_dataset = test_ava
elif args.VAL == 'AVAK':
    test_ava = AVA(args.AVA, args.ANNOTRAINAVA, args.ANNOVALAVA, args.ANNOLISTAVA2019, transforms=tfs, frames=args.FRAMES, rate=args.RATEAVA, split ='val')
    test_kinetics = K700(args.KINETICS, args.ANNOTRAINKINETICS, args.ANNOVALKINETICS, clsfile=args.ANNOLISTAVA2019, csvfile=args.ANNOLISTKINETICS, transforms=tfs, frames=args.FRAMES, rate=args.RATEKINETICS, split='val')
    test_dataset = torch.utils.data.ConcatDataset((test_ava, test_kinetics))
elif args.VAL == 'K700':
    test_ava = AVA(args.AVA, args.ANNOTRAINAVA, args.ANNOVALAVA, args.ANNOLISTAVA2019, transforms=tfs, frames=args.FRAMES, rate=args.RATEAVA, split ='val')
    test_kinetics = K700(args.KINETICS, args.ANNOTRAINKINETICS, args.ANNOVALKINETICS, clsfile=args.ANNOLISTAVA2019, csvfile=args.ANNOLISTKINETICS, transforms=tfs, frames=args.FRAMES, rate=args.RATEKINETICS, split='val')
    test_dataset = test_kinetics
    
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers = num_workers)

test_ucf = UCF24(args.UCF24, args.ANNOUCF24, transforms = tfs, frames = video_input_num_frames, rate = args.RATEUCF24, split = 'test')
test_ucf_loader = DataLoader(test_ucf, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers = num_workers)

test_hmdb = HMDB21(args.HMDB21, args.ANNOHMDB21, transforms = tfs, frames = video_input_num_frames, split = 'test')
test_hmdb_loader = DataLoader(test_hmdb, batch_size=1, collate_fn=collate_fn, shuffle=True, num_workers = 1)

def flatten(xss):
    return [x for xs in xss for x in xs]
    
# remove 20 long tail classes from AVA
ava60classes = test_ava.classes
avatextaug = dict(filter(lambda item: item[0] in ava60classes, avatextaug.items()))
print('AVA 60 classes:', list(avatextaug.keys()))

ava_captions = [k for k, v in avatextaug.items()]
ucf_captions = [k for k, v in ucf24textaug.items()]
hmdb_captions = [k for k, v in hmdb21textaug.items()]

if args.TXTAUG:
    ava_captions_aug = flatten([v for k, v in avatextaug.items()])
    ucf_captions_aug = flatten([v for k, v in ucf24textaug.items()])
    hmdb_captions_aug = flatten([v for k, v in hmdb21textaug.items()])

model.eval()

json.dump([], open(args.JSON, 'w'))

if args.TXTAUG:
    # averaging matrix
    Aava = torch.zeros(16*60, 60)
    for i in range(60):
        start, end = 16*i, 16*(i+1)
        Aava[start:end,i] = 1/16
    Aava = Aava.to(device) # (16x60, 60)

    Aucf = torch.zeros(16*24, 24)
    for i in range(24):
        start, end = 16*i, 16*(i+1)
        Aucf[start:end,i] = 1/16
    Aucf = Aucf.to(device) # (16x24, 24)

    Ajhmdb = torch.zeros(16*21, 21)
    for i in range(21):
        start, end = 16*i, 16*(i+1)
        Ajhmdb[start:end,i] = 1/16
    Ajhmdb = Ajhmdb.to(device) # (16x21, 21)
        
postprocess = PostProcess()

weights = os.listdir(args.PRETRAINED)
prefix = '_'.join(weights[0].split('_')[:-1])
weights = [int(ele.split('_')[-1].split('.')[0]) for ele in weights]
weights.sort()
weights = [prefix + '_' +str(ele) + '.pt' for ele in weights]
for weight in weights:
    print('evaluating:', weight)
    model.load_state_dict(torch.load(os.path.join(args.PRETRAINED, weight), map_location=device))
    model.eval()
    
    ############
    # Test AVA #
    ############
    metric = MeanAveragePrecision(iou_type='bbox',
                                  box_format='xyxy',
                                  iou_thresholds=[0.5, ],
                                  backend='faster_coco_eval')
    for samples, targets in tqdm(test_loader):
        with torch.no_grad():
            H, W = samples.shape[-2:]
            samples = samples.to(device)

            #extract captions and temp label mapping
            captions = ava_captions
            captionstoidx = {v: k for k, v in enumerate(captions)}
            temp_num_classes = len(captions)
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                intlabels = [list(map(lambda x:captionstoidx[x], ele)) for ele in t['text_labels']]
                t['labels'] = intlabels
            
            if args.TXTAUG:
                outputs = model(samples.to(device), ava_captions_aug)
                results = postprocess(outputs, imgsize=(H, W), human_conf=0.0, Aaug=Aava)
            else:
                outputs = model(samples.to(device), ava_captions)
                results = postprocess(outputs, imgsize=(H, W), human_conf=0.0)

        preds, tgts = [], []
        for i in range(samples.shape[0]):
            result = results[i]
            scores, cls, boxes = result['scores'], result['labels'], result['boxes']
            
            if boxes.shape[0] == 0:
                result_dict = None
            else:
                result_dict = {'boxes': boxes, 'labels': cls, 'scores': scores}
            
            gt = targets[i]
            rawboxes = gt['boxes']
            rawboxes = box_cxcywh_to_xyxy(rawboxes)
            rawcls = gt['labels']
            boxes = []
            cls = []
            for j in range(len(rawcls)):
                tmpcls = rawcls[j]
                cls.extend(tmpcls)
                for _ in range(len(tmpcls)):
                    boxes.append(rawboxes[j])
            boxes = torch.stack(boxes)
            boxes[:,0::2] = boxes[:,0::2] * W
            boxes[:,1::2] = boxes[:,1::2] * H
            cls = torch.tensor(cls).to(boxes.device)
            ground_truth = {'boxes': boxes, 'labels': cls}
            
            if result_dict != None:
                preds.append(result_dict)
                tgts.append(ground_truth)
        metric.update(preds, tgts)
        
    pprint(metric.compute())
    ava_fmap2, ava_fmap5 = 0, metric.compute()['map_50'].cpu().detach().item()

    ############
    # Test UCF #
    ############
    metric = MeanAveragePrecision(iou_type='bbox',
                                  box_format='xyxy',
                                  iou_thresholds=[0.5, ],
                                  backend='faster_coco_eval')
    for samples, targets in tqdm(test_ucf_loader):
        with torch.no_grad():
            samples = samples.to(device)

            #extract captions and temp label mapping
            captions = ucf_captions
            captionstoidx = {v: k for k, v in enumerate(captions)}
            temp_num_classes = len(captions)
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                intlabels = [list(map(lambda x:captionstoidx[x], ele)) for ele in t['text_labels']]
                t['labels'] = intlabels
            
            if args.TXTAUG:
                outputs = model(samples.to(device), ucf_captions_aug)
                results = postprocess(outputs, imgsize=test_ucf.imgsize, human_conf=0.0, Aaug=Aucf)
            else:
                outputs = model(samples.to(device), ucf_captions)
                results = postprocess(outputs, imgsize=test_ucf.imgsize, human_conf=0.0)

        preds, tgts = [], []
        for i in range(samples.shape[0]):
            result = results[i]
            scores, cls, boxes = result['scores'], result['labels'], result['boxes']
            
            if boxes.shape[0] == 0:
                result_dict = None
            else:
                result_dict = {'boxes': boxes, 'labels': cls, 'scores': scores}
            
            gt = targets[i]
            rawboxes = gt['boxes']
            rawboxes = box_cxcywh_to_xyxy(rawboxes)
            rawcls = gt['labels']
            boxes = []
            cls = []
            for j in range(len(rawcls)):
                tmpcls = rawcls[j]
                cls.extend(tmpcls)
                for _ in range(len(tmpcls)):
                    boxes.append(rawboxes[j])
            boxes = torch.stack(boxes)
            boxes[:,0::2] = boxes[:,0::2] * test_ucf.imgsize[1]
            boxes[:,1::2] = boxes[:,1::2] * test_ucf.imgsize[0]
            cls = torch.tensor(cls).to(boxes.device)
            ground_truth = {'boxes': boxes, 'labels': cls}
            
            if result_dict != None:
                preds.append(result_dict)
                tgts.append(ground_truth)
        metric.update(preds, tgts)
        
    pprint(metric.compute())
    ucf_fmap2, ucf_fmap5 = 0, metric.compute()['map_50'].cpu().detach().item()

    ##############
    # Test JHMDB #
    ##############
    metric = MeanAveragePrecision(iou_type='bbox',
                                  box_format='xyxy',
                                  iou_thresholds=[0.5, ],
                                  backend='faster_coco_eval')
    for samples, targets in tqdm(test_hmdb_loader):
        with torch.no_grad():
            samples = samples.to(device)

            #extract captions and temp label mapping
            captions = hmdb_captions
            captionstoidx = {v: k for k, v in enumerate(captions)}
            temp_num_classes = len(captions)
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                intlabels = [list(map(lambda x:captionstoidx[x], ele)) for ele in t['text_labels']]
                t['labels'] = intlabels
            
            if args.TXTAUG:
                outputs = model(samples.to(device), hmdb_captions_aug)
                results = postprocess(outputs, imgsize=test_hmdb.imgsize, human_conf=0.0, Aaug=Ajhmdb)
            else:
                outputs = model(samples.to(device), hmdb_captions)
                results = postprocess(outputs, imgsize=test_hmdb.imgsize, human_conf=0.0)

        preds, tgts = [], []
        for i in range(samples.shape[0]):
            result = results[i]
            scores, cls, boxes = result['scores'], result['labels'], result['boxes']
            
            if boxes.shape[0] == 0:
                result_dict = None
            else:
                result_dict = {'boxes': boxes, 'labels': cls, 'scores': scores}
            
            gt = targets[i]
            rawboxes = gt['boxes']
            rawboxes = box_cxcywh_to_xyxy(rawboxes)
            rawcls = gt['labels']
            boxes = []
            cls = []
            for j in range(len(rawcls)):
                tmpcls = rawcls[j]
                cls.extend(tmpcls)
                for _ in range(len(tmpcls)):
                    boxes.append(rawboxes[j])
            boxes = torch.stack(boxes)
            boxes[:,0::2] = boxes[:,0::2] * test_hmdb.imgsize[1]
            boxes[:,1::2] = boxes[:,1::2] * test_hmdb.imgsize[0]
            cls = torch.tensor(cls).to(boxes.device)
            ground_truth = {'boxes': boxes, 'labels': cls}
            
            if result_dict != None:
                preds.append(result_dict)
                tgts.append(ground_truth)
        metric.update(preds, tgts)
        
    pprint(metric.compute())
    hmdb_fmap2, hmdb_fmap5 = 0, metric.compute()['map_50'].cpu().detach().item()

    stats = json.load(open(args.JSON))
    stats.append({'ava0.2': ava_fmap2,
                  'ava0.5': ava_fmap5,
                  'ucf0.2': ucf_fmap2,
                  'ucf0.5': ucf_fmap5,
                  'jhmdb0.2': hmdb_fmap2,
                  'jhmdb0.5': hmdb_fmap5})
    json.dump(stats, open(args.JSON, 'w'))
