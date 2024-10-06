import os
import clip
import torch
import pickle
from tqdm import tqdm
import numpy as np
import pyarrow.parquet as pq

device = "cuda:3" if torch.cuda.is_available() else "cpu"
model, _, preprocess = clip.load('./~/.cache/clip/RN50.pt', device)

with open('./train_new_label/test_01_coco_925.pkl', 'rb') as f:
    coco_data = pickle.load(f)

with open('./train_new_label/test_01_nuswide_925.pkl', 'rb') as f:
    nuswide_data = pickle.load(f)

all_data = coco_data[0] + nuswide_data[0]

cls2coco = {}
coco_classname_synonyms = [
    ['person', 'human', 'people', 'man', 'woman', 'family','boy', 'girl', 'passenger', 'child', 'adult', 'elderly', 'youth', 'citizen', 'individual', 'worker', 'student', 'teacher', 'doctor', 'engineer', 'scientist', 'artist', 'athlete', 'musician', 'actor', 'actress', 'performer', 'customer', 'tourist', 'resident', 'family', 'friend', 'neighbor', 'stranger', 'colleague', 'boss', 'employee', 'team member', 'leader', 'voter', 'patient', 'passerby', 'pedestrian', 'commuter', 'shopper', 'traveler', 'pilgrim', 'bystander', 'singer', 'dancer', 'chef', 'waiter', 'waitress', 'bartender', 'nurse', 'lawyer', 'judge', 'policeman', 'policewoman', 'firefighter', 'soldier', 'veteran', 'ambassador', 'diplomat', 'journalist', 'reporter', 'photographer', 'writer', 'author'],
    ['bicycle', 'bike', 'cycle'],
    ['car', 'taxi', 'auto', 'automobile', 'motor car'], 
    ['motorcycle', 'motor bike', 'motor cycle'], 
    ['airplane', 'aeroplane', "air craft", "jet", "plane", "air plane"], 
    ['bus', 'autobus', 'coach', 'charabanc', 'double decker', 'jitney', 'motor bus', 'motor coach', 'omnibus'],
    ['train', 'rail way', 'railroad'], 
    ['truck'],
    ['boat', 'raft', 'dinghy'],
    ['traffic light'],
    ['fire hydrant', 'fire tap', 'hydrant'],
    ['stop sign', 'halt sign'],
    ['parking meter'],
    ['bench'],
    ['bird'],
    ['cat', 'kitty'],
    ['dog', 'pup', 'puppy', 'doggy'],
    ['horse', 'colt', 'equus'],
    ['sheep'],
    ['cow'],
    ['elephant'],
    ['bear'],
    ['zebra'],
    ['giraffe', 'camelopard'],
    ['backpack', 'back pack', 'knapsack', 'packsack', 'rucksack', 'haversack'],
    ['umbrella'],
    ['handbag', 'hand bag', 'pocketbook', 'purse', 'bag'],
    ['tie', 'necktie'],
    ['suitcase'],
    ['frisbee'],
    ['skis', 'ski'],
    ['snowboard'],
    ['sports ball', 'sport ball', 'ball', 'football', 'soccer', 'tennis', 'basketball', 'baseball'],
    ['kite'],
    ['baseball bat', 'baseball game bat'],
    ['baseball glove', 'baseball mitt', 'baseball game glove'],
    ['skateboard'],
    ['surfboard'],
    ['tennis racket'],
    ['bottle'],
    ['wine glass', 'vino glass'],
    ['cup'],
    ['fork'],
    ['knife'],
    ['spoon'],
    ['bowl'],
    ['banana'],
    ['apple'],
    ['sandwich'],
    ['orange'],
    ['broccoli'],
    ['carrot'],
    ['hot dog'],
    ['pizza'],
    ['donut', 'doughnut'],
    ['cake'],
    ['chair', 'arm chair'],
    ['couch', 'sofa'],
    ['potted plant', 'house plant', 'bonsai', 'pot plant'],
    ['bed'],
    ['dining table', 'dinner table', 'table', 'din table'], 
    ['toilet', 'commode'],
    ['tv', 'tvmonitor', 'monitor', 'television', 'telly'],
    ['laptop'],
    ['mouse'],
    ['remote', 'remote control'],
    ['keyboard', 'typing board'],
    ['cell phone', 'phone', 'mobile phone', 'telephone'],
    ['microwave'],
    ['oven', 'roaster'],
    ['toaster', 'bread-making machine'],
    ['sink', 'basin', 'washbasin'],
    ['refrigerator', 'icebox'],
    ['book','magazine'],
    ['clock','timepiece'],
    ['vase', 'flower holder'],
    ['scissors'],
    ['teddy bear', 'teddy', 'teddie bear', 'toy bear'],
    ['hair drier', 'blowing machine', 'hair dryer', 'dryer', 'blow dryer', 'blown dry', 'blow dry', 'hair dry'],
    ['toothbrush'],
]
for c in coco_classname_synonyms:
    for cc in c:
        cls2coco[cc] = c[0]
        cc = cc.replace(' ', '')
        cls2coco[cc] = c[0]
        
cls_list = []
caption_list = []
embed_list = []

# all_data = all_data[:100]
step = 32
for idx in tqdm(range(0, len(all_data), step)):
    data = all_data[idx:idx+step]
    cls_cap = [t.split('&&') for t in data]
    # print(cls_cap)
    text_inputs = torch.cat([clip.tokenize(c[0]) for c in cls_cap]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu().numpy()
    
    for i in range(len(cls_cap)):
        
        new_cls = []
        # print(cls_cap[i][1].split(','))
        for x in cls_cap[i][1].split(','):
            if x in cls2coco:
                new_cls.append(cls2coco[x])
        if new_cls:
            cls_list.append(new_cls)
            caption_list.append(cls_cap[i][0])
            embed_list.append(text_features[i])
            
print(f"number of text data: {len(cls_list)}")
            
with open('text_data/rn50/coco_cls_captions_embed.pkl', 'wb') as f:
    pickle.dump([cls_list, caption_list, embed_list], f)