'''
Code for loading DinoV2Salad as a global features extractor
for geolocalization through image retrieval.

DinoV2Salad paper: https://arxiv.org/abs/2311.15937
'''


import torch
from ..utils.base_model import BaseModel


class DinoV2Salad(BaseModel):
    default_conf = {
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.net = torch.hub.load(
            "serizba/salad", 
            "dinov2_salad",
            **conf
        ).eval()


    def _forward(self, data):
        desc = self.net(data['image'])
        return {
            'global_descriptor': desc,
        }
