import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image
import decord
from decord import VideoReader
import numpy as np
import os
import json
import regex as re
import random
import pandas as pd
import random

from util.box_ops import box_xyxy_to_cxcywh

decord.bridge.set_bridge('torch')

novel15 = {80: "watch (a person)",
           12: "stand",
           17: "carry/hold (an object)",
           4: "dance",
           61: "watch (e.g., TV)",
           38: "open (e.g., a window, a car door)",
           70: "hug (a person)",
           37: "listen (e.g., to music)",
           72: "kiss (a person)",
           26: "dress/put on clothing",
           76: "push (another person)",
           5: "fall down",
           52: "shoot",
           34: "hit (an object)",
           24: "cut"}

class AVA(Dataset):
    def __init__(self, video_dir,
                 train_csv='ava_train_v2.2.csv',
                 val_csv='ava_val_v2.2.csv',
                 clsfile='ava_action_list_v2.2.pbtxt',
                 transforms = None,
                 frames = 8,
                 rate = 8,
                 mode = 'video',
                 split='train'):
        assert split in ('train', 'val', 'base', 'novel'), 'split must be either train, val, base or novel'
        assert mode in ('frame', 'video'), 'mode must be either video or frames'
        self.split = split
        self.mode = mode
        self.video_dir = video_dir
        self.frames = frames
        self.rate = rate
        self.T = frames * rate
        self.transforms = transforms

        self.fps = 30 # AVA videos are postprocessed to 30 fps following SlowFast
        
        # Get idx-to-text mapping
        if split in ('train',):
            self.idxtolabel = self._extractidxtolabel(clsfile)
        else:
            self.idxtolabel = self._extractidxtolabel2019(clsfile)
            
        self.classes = [v for k,v in self.idxtolabel.items()]
        if split == 'base':
            novel_classes = [v for k, v in novel15.items()]
            self.classes = list(filter(lambda x: x not in novel_classes, self.classes))
        elif split == 'novel':
            novel_classes = [v for k, v in novel15.items()]
            self.classes = novel_classes

        # Process bboxes per clip
        if split == 'train':
            self.video_frame_bbox, self.frame_keys_list = self._obtain_generated_bboxes_training(train_csv)
        elif split == 'val':
            self.video_frame_bbox, self.frame_keys_list = self._obtain_generated_bboxes_training(val_csv)
        elif split == 'base':
            self.video_frame_bbox, self.frame_keys_list = self._obtain_generated_bboxes_training(train_csv, extra = val_csv)
        elif split == 'novel':
            self.video_frame_bbox, self.frame_keys_list = self._obtain_generated_bboxes_training(val_csv)
        
        self.suffix = ('.mp4', '.mkv', '.webm')

    def __len__(self):
        return len(self.frame_keys_list)

    def __getitem__(self, idx):
        out_dict = {}
        vt = self.frame_keys_list[idx]
        target = self.video_frame_bbox[vt]

        # extract text labels per bbox
        intlabels = target['acts']
        textlabels = []
        textlabels = [list(map(lambda x: self.idxtolabel.get(x), ele)) for ele in intlabels]
        out_dict['text_labels'] = textlabels

        # load video
        vid, t_second = vt.split(",")
        midframe = (int(t_second) - 900) * self.fps #minus 900 seconds because videos are cropped from 15min to 30min
        startframe = midframe - self.rate * (self.frames // 2)
        
        clip_idx = list(range(startframe, startframe + self.T, self.rate)) # start from 0 or 1?  check TubeR
        clip_idx = list(map(lambda x: 0 if x < 0 else x, clip_idx))
        
        clip = self._loadclip(vid, clip_idx) / 255

        H, W = clip.shape[2:]
        
        # augmentations
        bboxes = torch.tensor(target['bboxes']) #normalized, convert back to raw pixel coords
        bboxes[:, 0::2] = bboxes[:,0::2] * W
        bboxes[:, 1::2] = bboxes[:,1::2] * H
        bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=(H,W))

        if self.transforms != None:
            clip, bboxes = self.transforms(clip, bboxes)
            #clip = self.transforms(clip)

        canvas_size = bboxes.canvas_size
        # XYXY to CXCYWH
        bboxes = box_xyxy_to_cxcywh(bboxes)

        # raw to normal
        bboxes[:, 0::2] = bboxes[:,0::2] / canvas_size[1]
        bboxes[:, 1::2] = bboxes[:,1::2] / canvas_size[0]

        out_dict['boxes'] = bboxes
        
        out_dict['keyframe'] = midframe
        out_dict['video'] = vid

        return clip, out_dict
        
    def _loadclip(self, vid, clip_idx):
        if self.mode == 'frame':
            vidlen = len(os.listdir(os.path.join(self.video_dir, vid)))
            clip_idx = list(map(lambda x: vidlen if x > vidlen else x, clip_idx))
            
            clip = [os.path.join(self.video_dir, vid, vid+'_'+str(idx + 1).zfill(5)+'.jpg') for idx in clip_idx]
            clip = [read_image(pth) for pth in clip]
            clip = torch.stack(clip)
        else:
            v_pth = os.path.join(self.video_dir, vid)
            for s in self.suffix:
                if os.path.exists(v_pth + s):
                    v_pth  = v_pth + s
                    break
            vr = VideoReader(os.path.join(v_pth))
            
            vidlen = len(vr)
            clip_idx = list(map(lambda x: vidlen if x >= vidlen else x, clip_idx))
            clip = vr.get_batch(clip_idx).permute(0,3,1,2)
        return clip

    def _extractidxtolabel(self, clsfile):
        f = open(clsfile)
        idxtolabel = {}
        i = 0
        for line in f.readlines():
            if i == 1:
                text_label = line[:-2][9:]
            elif i == 2:
                label = int(line[:-1][12:])
            i += 1
            if i == 5:
                idxtolabel[label] = text_label
                i = 0
        f.close()
        return idxtolabel
        
    def _extractidxtolabel2019(self, clsfile):
        f = open(clsfile)
        idxtolabel = {}
        i = 0
        for line in f.readlines():
            if i == 1:
                text_label = line[:-2][9:]
            elif i == 2:
                label = int(line[:-1][6:])
            i += 1
            if i == 4:
                idxtolabel[label] = text_label
                i = 0
        f.close()
        return idxtolabel

    def _obtain_generated_bboxes_training(self, csv, extra = None):
        used=[]
    
        video_frame_bbox = {}
        gt_sheet = pd.read_csv(csv, header=None)
        gt_sheet = gt_sheet.dropna()
        if extra != None:
            extra_sheet = pd.read_csv(csv, header=None)
            extra_sheet = extra_sheet.dropna()
            
            gt_sheet = pd.concat([gt_sheet, extra_sheet])
        count = 0
        frame_keys_list = set()
        missed_videos = set()
    
        for index, row in gt_sheet.iterrows():
            vid = row[0]
    
            frame_second = row[1]
    
            bbox_conf = row[7]
            if bbox_conf < 0.8:
                continue
            frame_key = "{},{}".format(vid, str(frame_second).zfill(4))
    
            count += 1
            bbox = [row[2], row[3], row[4], row[5]]
            gt = int(row[6])
            
            if gt not in self.idxtolabel: # for base2novel, using 60 classes based on activitynet2019
                continue
                
            if self.split == 'base' and gt in novel15:
                continue
            elif self.split == 'novel' and gt not in novel15:
                continue
                
            frame_keys_list.add(frame_key)
    
            if frame_key not in video_frame_bbox.keys():
                video_frame_bbox[frame_key] = {}
                video_frame_bbox[frame_key]["bboxes"] = [bbox]
                video_frame_bbox[frame_key]["acts"] = [[gt]]
            else:
                if bbox not in video_frame_bbox[frame_key]["bboxes"]:
                    video_frame_bbox[frame_key]["bboxes"].append(bbox)
                    video_frame_bbox[frame_key]["acts"].append([gt])
                else:
                    idx = video_frame_bbox[frame_key]["bboxes"].index(bbox)
                    video_frame_bbox[frame_key]["acts"][idx].append(gt)
        
        return video_frame_bbox, list(frame_keys_list)
        
avatextaug = json.load(open('gpt/GPT_AVA.json'))

def textaugava(text_lst):
    aug_text_lst = []
    for text in text_lst:
        if text not in avatextaug:
            raise Exception('label must be in AVA')
        else:
            aug_text_lst.append(random.choice(avatextaug[text]))
    return aug_text_lst

k700textaug = json.load(open('gpt/GPT_K700.json'))

def textaugavak(text_lst):
    aug_text_lst = []
    for text in text_lst:
        if text not in avatextaug and text not in k700textaug:
            msg = 'label must be in AVA or K700, ' + text + ' is invalid'
            raise Exception(msg)
        elif text in avatextaug:
            aug_text_lst.append(random.choice(avatextaug[text]))
        elif text in k700textaug:
            aug_text_lst.append(random.choice(k700textaug[text]))
    return aug_text_lst
