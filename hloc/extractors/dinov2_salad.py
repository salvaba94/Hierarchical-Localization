'''
Code for loading DinoV2Salad as a global features extractor
for geolocalization through image retrieval.

DinoV2Salad paper: https://arxiv.org/abs/2311.15937
'''


import torch
import torchvision.transforms as T

from ..utils.base_model import BaseModel


class DinoV2Salad(BaseModel):
    default_conf = {
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.net = torch.hub.load(
            "serizba/salad", 
            "dinov2_salad"
        ).eval()

        self.transform = T.Compose([
            T.Resize((322, 322),  interpolation=T.InterpolationMode.BILINEAR),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def _forward(self, data):
        image = self.transform(data['image'])
        desc = self.net(image)
        return {
            'global_descriptor': desc,
        }
