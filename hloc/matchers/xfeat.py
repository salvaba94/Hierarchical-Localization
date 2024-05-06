'''
Code for loading XFeat as a global features extractor.

XFeat paper: https://arxiv.org/abs/2404.19174
'''
from types import SimpleNamespace

import torch

from ..utils.base_model import BaseModel


class XFeatMatcher(BaseModel):
    default_conf = {
        'top_k' : 4096,
        'semi_dense' : True
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    def _init(self, conf):
        self.semi_dense = conf.pop('semi_dense')
        self.multi_scale = conf.pop('multi_scale')
        self.net = torch.hub.load(
            'verlab/accelerated_features', 
            'XFeat',
            pretrained = True, 
            top_k=conf["top_k"]
        ).eval()
        

    def _forward(self, data):
        if not self.semi_dense:
            preds = self.net.match_xfeat(data['image0'], data['image1'])
        else:
            preds = self.net.match_xfeat_star(data['image0'], data['image1'])
        return preds
