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

from tqdm import tqdm

from util.box_ops import box_xyxy_to_cxcywh

decord.bridge.set_bridge('torch')

class MultiSports(Dataset):
    def __init__(self, video_dir, anno_pth,
                 transforms = None,
                 frames = 9,
                 rate = 8,
                 split = 'test'):
        self.video_dir = video_dir
        self.anno = pkl.load(open(anno_pth, 'rb'), encoding='latin1')
        self.gttubes = self.anno['gttubes']
        self.nframes = self.anno['nframes']
        self.resolution = self.anno['resolution']
        
        self.transforms = transforms
        self.frames = frames
        self.rate = rate
        self.T = frames * rate
    
        if split == 'train':
            self.video_list = self.anno['train_videos'][0]
            self.classes = self.anno['labels']
            #self.classes = list(map(lambda x:' '.join(re.findall('[A-Z][a-z]*', x)), self.classes))
        elif split == 'test':
            self.video_list = self.anno['test_videos'][0]
            self.classes = self.anno['labels']
            #self.classes = list(map(lambda x:' '.join(re.findall('[A-Z][a-z]*', x)), self.classes))
    
        self.bboxes = {}
        for idx in range(len(self.video_list)):
            # get bboxes per frame
            filepth = self.video_list[idx]
            gttubes = self.gttubes[self.video_list[idx]]
            for action in gttubes:
                gttube_set = gttubes[action]
                for tube in gttube_set:
                    for bbox in tube:
                        img_id, x1, y1, x2, y2 = bbox
                        img_id, x1, y1, x2, y2 = int(img_id), int(x1), int(y1), int(x2), int(y2)
                        
                        file_dir = os.path.join(video_dir, filepth)
                        if file_dir not in self.bboxes:
                            self.bboxes[file_dir] = {}
                        
                        if img_id not in self.bboxes[file_dir]:
                            self.bboxes[file_dir][img_id] = [torch.tensor([action, x1,y1,x2,y2])]
                        else:
                            self.bboxes[file_dir][img_id].append(torch.tensor([action, x1,y1,x2,y2]))

        self.indices = []
        if str(frames)+'x'+str(rate)+'_'+split + 'indices_multisports.json' in os.listdir('/'.join(anno_pth.split('/')[:-1])):
            self.indices = json.load(open(os.path.join('/'.join(anno_pth.split('/')[:-1]), str(frames)+'x'+str(rate)+'_'+split + 'indices_multisports.json')))
        else:
            for v in tqdm(self.video_list):
                vtubes = sum(self.gttubes[v].values(), [])
                self.indices += [(v, i) for i in range(1, self.nframes[v] + 2 - self.T) if tubelet_in_out_tubes(vtubes, i, self.T) and tubelet_has_gt(vtubes, i, self.T)]
            json.dump(self.indices, open(os.path.join('/'.join(anno_pth.split('/')[:-1]), str(frames)+'x'+str(rate)+'_'+split + 'indices_multisports.json'), 'w'))

        print(split, 'dataset (multisports) loaded!')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        v, frame = self.indices[idx]
        h, w = self.resolution[v]
        v = os.path.join(self.video_dir, v)
        vr = VideoReader(v + '.mp4')

        #label = v.split('/')[-2] # no global labels
        
        # sample random segment from video
        start = frame - 1
        end = start + self.T
        clip_idx = list(range(start, end, self.rate))

        # load clip
        clip = vr.get_batch(clip_idx).permute(0, 3, 1, 2) / 255 # TCHW, scaled between 0 and 255

        # load anno
        mididx = clip_idx[self.frames // 2] + 1
        try:
            bboxes = self.bboxes[v][mididx]
        except KeyError:
            bboxes = None

        if bboxes != None:
            bboxes = torch.stack(bboxes).float()
            labels, bboxes = bboxes[:, 0], bboxes[:, 1:]
            bboxes[:, 0::2].clamp_(min=0, max=w)
            bboxes[:, 1::2].clamp_(min=0, max=h)
        else:
            labels = [[]]
            bboxes = torch.empty((0, 4), dtype=torch.float32)

        bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=(h,w))

        if self.transforms != None:
            clip, bboxes = self.transforms(clip, bboxes)
            #clip = self.transforms(clip)

        canvas_size = bboxes.canvas_size
        # XYXY to CXCYWH
        bboxes = box_xyxy_to_cxcywh(bboxes)

        # raw to normal
        bboxes[:, 0::2] = bboxes[:,0::2] / canvas_size[1]
        bboxes[:, 1::2] = bboxes[:,1::2] / canvas_size[0]

        # process labels
        labels = [[self.classes[int(ele)]] for ele in labels]

        target = {'boxes': bboxes,
                  'text_labels': labels,
                  'keyframe': mididx,
                  'video': v}
        
        return clip, target

def tubelet_in_tube(tube, i, K):
    # True if all frames from i to (i + K - 1) are inside tube
    # it's sufficient to just check the first and last frame.
    # return (i in tube[: ,0] and i + K - 1 in tube[:, 0])
    return all([j in tube[:, 0] for j in range(i, i + K)])


def tubelet_out_tube(tube, i, K):
    # True if all frames between i and (i + K - 1) are outside of tube
    return all([not j in tube[:, 0] for j in range(i, i + K)])


def tubelet_in_out_tubes(tube_list, i, K):
    # Given a list of tubes: tube_list, return True if
    # all frames from i to (i + K - 1) are either inside (tubelet_in_tube)
    # or outside (tubelet_out_tube) the tubes.
    return all([tubelet_in_tube(tube, i, K) or tubelet_out_tube(tube, i, K) for tube in tube_list])

def tubelet_has_gt(tube_list, i, K):
    # Given a list of tubes: tube_list, return True if
    # the tubelet starting spanning from [i to (i + K - 1)]
    # is inside (tubelet_in_tube) at least a tube in tube_list.
    return any([tubelet_in_tube(tube, i, K) for tube in tube_list])

multisportstextaug = json.load(open('gpt/GPT_MultiSports.json'))

def textaugmultisports(text_lst):
    aug_text_lst = []
    for text in text_lst:
        if text not in multisportstextaug:
            aug_text_lst.append(text)
        else:
            aug_text_lst.append(random.choice(multisportstextaug[text]))
    return aug_text_lst
