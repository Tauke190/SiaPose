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

# NOTE: The Kinetics timestamp annotations in AVA-Kinetics do not
#       correspond to a 1-to-1 mapping with the timestamps of the
#       original K700 10-second videos. Preprocessing is required
#       to use the Kinetics annotations of AVA-Kinetics, which is
#       included in the nn.Dataset class below.

errlst = json.load(open('datasets/corrupted_train.json')) # temporary fix to remove corrupted k700 videos

class K700(Dataset):
    def __init__(self, video_dir,
                 train_csv='kinetics_train_v1.0.csv',
                 val_csv='kinetics_val_v1.0.csv',
                 clsfile='ava_action_list_v2.2.pbtxt',
                 csvfile='kinetics_700_labels.csv',
                 transforms = None,
                 frames = 8,
                 rate = 8,
                 original_timestamp_csv = None,
                 split='train',
                 enable_k700_labels=False,
                 aligned_anno='',
                 corrupted_vids = errlst):
        assert split in('train', 'val'), 'split must either be train or val'
        self.corrupted_vids = corrupted_vids
        self.split = split
        self.enable_k700_labels = enable_k700_labels
        self.video_dir = video_dir
        self.frames = frames
        self.rate = rate
        self.transforms = transforms

        self.fps = 30 # Kinetics videos have varying fps, might wanna reconsider using variable sampling rate

        # Get idx-to-text mapping
        if split in ('train',):
            self.idxtolabelavak = self._extractidxtolabelAVAK(clsfile)
        else:
            self.idxtolabelavak = self._extractidxtolabelAVAK2019(clsfile)
        selfidxtolabelk700 = self._extractidxtolabelK700(csvfile)
        
        # Process bboxes per clip
        if split == 'train':
            self.video_frame_bbox, self.frame_keys_list, self.filemap = self._obtain_generated_bboxes_training(train_csv)
        else:
            self.video_frame_bbox, self.frame_keys_list, self.filemap = self._obtain_generated_bboxes_training(val_csv)
            
        # (optional) add aligned weak-supervision annotations
        self.aligned_anno = None
        if aligned_anno != '':
            self.aligned_anno = json.load(open(aligned_anno))
        
    def __len__(self):
        return len(self.frame_keys_list)
        
    def __getitem__(self, idx):
        out_dict = {}
        vt = self.frame_keys_list[idx]
        target = self.video_frame_bbox[vt]
        
        # load video
        vid, t_second = vt.split(",")
        clip, global_cls, midframe, v_pth = self._loadclip(vid, t_second)
        clip = clip / 255
        H, W = clip.shape[2:]

        # extract text labels per bbox
        intlabels = target['acts']
        textlabels = [list(map(lambda x: self.idxtolabelavak.get(x), ele)) for ele in intlabels]
        if self.enable_k700_labels and self.split == 'train':
            if self.aligned_anno is not None:
                tgtidx = self.aligned_anno[vt]
                textlabels = [ele + [global_cls] if i in tgtidx else ele for i, ele in enumerate(textlabels)]
            else:
                textlabels = [ele + [global_cls] for ele in textlabels]
        out_dict['text_labels'] = textlabels
        
        # augmentations
        bboxes = torch.tensor(target['bboxes']).float() #normalized, convert back to raw pixel coords, MUST USE float() TO CONVERT TO FLOAT32 FROM 64
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
        
        #out_dict['keyframe'] = midframe
        #out_dict['video'] = v_pth
        out_dict['global_cls'] = global_cls
        out_dict['frame_key'] = vt

        return clip, out_dict
        
    def _loadclip(self, vid, t_second):
        v_pth = self.filemap[vid]
        global_cls = v_pth.split('/')[-2]
        vr = VideoReader(v_pth)
        
        vidlen = len(vr)
        actualfps = vr.get_avg_fps()
        midframe = int(int(t_second) * actualfps)
        
        newrate = int(self.rate * actualfps / self.fps) # variable sampling rate based on video fps
        startframe = midframe - newrate * (self.frames // 2)
        clip_idx = list(range(startframe, startframe + self.frames * newrate, newrate)) # start from 0 or 1?  check TubeR
        clip_idx = list(map(lambda x: 0 if x < 0 else x, clip_idx))
        clip_idx = list(map(lambda x: vidlen - 1 if x >= vidlen else x, clip_idx))
        clip = vr.get_batch(clip_idx).permute(0,3,1,2)
        return clip, global_cls, midframe, v_pth
        
    def _getoriginaltimestamps(self):
        d = {}
        filemap = {}
        
        trainfolder = os.path.join(self.video_dir, 'train')
        for cls in os.listdir(trainfolder):
            clstrainfolder = os.path.join(trainfolder, cls)
            video_list = os.listdir(clstrainfolder)
            for video in video_list:
                vidid = video[0:11]
                start = int(video[12:18])
                end = int(video[19:25])
                d[vidid] = (start, end)
                filemap[vidid] = os.path.join(clstrainfolder, video)
        
        valfolder = os.path.join(self.video_dir, 'val')
        for cls in os.listdir(valfolder):
            clsvalfolder = os.path.join(valfolder, cls)
            video_list = os.listdir(clsvalfolder)
            for video in video_list:
                vidid = video[0:11]
                start = int(video[12:18])
                end = int(video[19:25])
                d[vidid] = (start, end)
                d[vidid] = (start, end)
                filemap[vidid] = os.path.join(clsvalfolder, video)
        return d, filemap
        
    def _obtain_generated_bboxes_training(self, csv):
        original_timestamps, filemap = self._getoriginaltimestamps()
        
        video_frame_bbox = {}
        frame_keys_list = set()
        gt_sheet = pd.read_csv(csv, names=(0,1,2,3,4,5,6))
        gt_sheet = gt_sheet.dropna()
        
        for i in range(len(gt_sheet)):
            vidid = gt_sheet.iloc[i,0]
            key = int(gt_sheet.iloc[i,1])
            x1 = gt_sheet.iloc[i,2]
            y1 = gt_sheet.iloc[i,3]
            x2 = gt_sheet.iloc[i,4]
            y2 = gt_sheet.iloc[i,5]
            act = gt_sheet.iloc[i,6]
            
            if vidid not in original_timestamps:
                continue
                
            if vidid in self.corrupted_vids:
                continue
                
            if act not in self.idxtolabelavak: # for val, using 60 classes based on activitynet2019
                continue
            
            start, end = original_timestamps[vidid]
            if start < key < end:
                newstart = key - start
                frame_key = "{},{}".format(vidid, str(newstart).zfill(4))
                frame_keys_list.add(frame_key)
                bbox = [x1, y1, x2, y2]
                gt = act
                
                if frame_key not in video_frame_bbox.keys():
                    video_frame_bbox[frame_key] = {}
                    video_frame_bbox[frame_key]["bboxes"] = [bbox]
                    video_frame_bbox[frame_key]["acts"] = [[gt]]
                    # add original k700 cls here
                else:
                    if bbox not in video_frame_bbox[frame_key]["bboxes"]:
                        video_frame_bbox[frame_key]["bboxes"].append(bbox)
                        video_frame_bbox[frame_key]["acts"].append([gt])
                    else:
                        idx = video_frame_bbox[frame_key]["bboxes"].index(bbox)
                        video_frame_bbox[frame_key]["acts"][idx].append(gt)
                        
        return video_frame_bbox, list(frame_keys_list), filemap
                 
    def _extractidxtolabelAVAK(self, clsfile):
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
        
    def _extractidxtolabelAVAK2019(self, clsfile):
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
        
    def _extractidxtolabelK700(self, csvfile):
        d = {}
        anno = pd.read_csv(csvfile)
        for i in range(len(anno)):
            vidid = anno.iloc[i,0]
            name = anno.iloc[i,1]
            d[vidid] = name
        return d
