'''
Code for loading XFeat as a global features extractor.

XFeat paper: https://arxiv.org/abs/2404.19174
'''

import torch

from ..utils.base_model import BaseModel

class XFeat(BaseModel):
    default_conf = {
        'top_k' : 4096
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.model = torch.hub.load(
            'verlab/accelerated_features', 
            'XFeat',
            pretrained = True, 
            top_k=conf["top_k"]
        ).eval()
        

    def _forward(self, data):
        self.model.parse_input(data["image"])
        preds = self.model.detectAndComputeDense(data['image'])
        preds["scores"] = torch.zeros_like(preds["scales"]) # No scores at the moment
        return preds
