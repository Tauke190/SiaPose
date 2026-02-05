import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
import numpy as np
import pickle as pkl
import os
import decord
from decord import VideoReader
import json
import regex as re
import random

from util.box_ops import box_xyxy_to_cxcywh

decord.bridge.set_bridge('torch')

nohuman = ['Vehicle UTurn',
           'Vehicle PicksUp Person',
           'Vehicle Starting',
           'Vehicle Turning Left',
           'Vehicle Stopping',
           'Vehicle DropsOff Person',
           'Vehicle Reversing',
           'Vehicle Turning Right']

class UCFMAMA(Dataset):
    def __init__(self, video_dir, anno_cache_dir, clsfile,
                 transforms = None,
                 imgsize = (1080,1920),
                 frames = 9,
                 rate = 4,
                 split = 'train'):
        assert split in ('train', 'test'), 'split must be train or test'
        self.video_dir = video_dir
        self.split = split
        self.split_dir = os.path.join(video_dir, split)
        
        self.transforms = transforms
        self.imgsize = imgsize
        self.frames = frames
        self.rate = rate
        self.T = frames * rate
        
        self.fps = 30 # to my knowledge, VIRAT videos are 30fps, not sure about MEVA
        
        self.idxtolabel = self._extractidxtolabel(clsfile)
        self.classes = [v.replace('_', ' ') for k, v in self.idxtolabel.items() if v.replace('_', ' ') not in nohuman]
        
        # process annotations
        if str(frames)+'x'+str(rate)+'_'+split +'indices_ucfmama.json' in os.listdir(anno_cache_dir):
            self.video_list = json.load(open(os.path.join(anno_cache_dir, str(frames)+'x'+str(rate)+'_'+split +'indices_ucfmama.json')))
        else:
            self.video_list = {}
            for vid in os.listdir(self.split_dir):
                anno = json.load(open(os.path.join(self.split_dir, vid, vid + '.json')))
                
                for act_id in anno:
                    subanno = anno[act_id]
                    action = subanno['action_type'].replace('_', ' ') # do I really need this during preprocessing?
                    
                    if action in nohuman: # remove non-human bboxes
                        continue
                    
                    vidlen = subanno['end_frame']
                    if vidlen < self.T:
                        continue
                    
                    for frameid in subanno['detections']:
                        # ignore if start and end exceed video clip
                        if int(frameid) - self.rate * (self.frames // 2) or int(frameid) + self.rate * (self.frames // 2 + 1) >= vidlen:
                            continue
                    
                        detections = subanno['detections'][frameid]
                        bboxes = [v for k, v in detections.items()]
                        
                        if vid not in self.video_list:
                            self.video_list[vid] = {}
                        
                        if frameid not in self.video_list[vid]:
                            self.video_list[vid][frameid] = {}
                            self.video_list[vid][frameid]['bboxes'] = bboxes
                            self.video_list[vid][frameid]['act'] = [[action]] * len(bboxes)
                        else:
                            self.video_list[vid][frameid]['bboxes'].extend(bboxes)
                            self.video_list[vid][frameid]['act'].extend([[action]] * len(bboxes))
            json.dump(self.video_list, open(os.path.join(anno_cache_dir, str(frames)+'x'+str(rate)+'_'+split +'indices_ucfmama.json'), 'w'))
            
        self.video_indices = []
        for v in self.video_list:
            for keyframe in self.video_list[v]:
                self.video_indices.append((v, keyframe))
        
    def __len__(self):
        return len(self.video_indices)
        
    def __getitem__(self, idx):
        video, keyframe = self.video_indices[idx]
        video_anno = self.video_list[video][keyframe]
        video_pth = os.path.join(self.video_dir, self.split, video, video + '.mp4')
        vr = VideoReader(video_pth)
        
        # load video
        keyframe = int(keyframe)
        start = keyframe - self.rate * (self.frames // 2)
        end = keyframe + self.rate * (self.frames // 2 + 1)
        indices = list(range(start, end, self.rate))
        
        clip = vr.get_batch(indices).permute(0,3,1,2) / 255
        
        H, W = clip.shape[2:]
        
        bboxes = torch.tensor(video_anno['bboxes'])
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
        
        textlabels = video_anno['act']
        
        out_dict = {'boxes': bboxes, 'text_labels': textlabels}
        
        return clip, out_dict
        
    def _extractidxtolabel(self, clsfile):
        f = open(clsfile)
        idxtolabel = {}
        for line in f.readlines():
            idx, label = line.split(' ')
            idx = int(idx)
            idxtolabel[idx] = label[:-1]
        f.close()
        return idxtolabel

ucfmamatextaug = json.load(open('gpt/GPT_UCFMAMA.json'))

def textaugucfmama(text_lst):
    aug_text_lst = []
    for text in text_lst:
        if text not in ucfmamatextaug:
            aug_text_lst.append(text)
        else:
            aug_text_lst.append(random.choice(ucfmamatextaug[text]))
    return aug_text_lst
