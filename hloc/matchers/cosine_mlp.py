'''
Code for loading XFeat as a global features extractor.

XFeat paper: https://arxiv.org/abs/2404.19174
'''

import torch

from ..utils.base_model import BaseModel


class CosineMLP(BaseModel):
    default_conf = {
        'top_k' : 4096,
        'semi_dense' : True,
        'min_cossim' : -1 
    }
    required_inputs = [
        "image0",
        "keypoints0",
        "descriptors0",
        "image1",
        "keypoints1",
        "descriptors1",
    ]

    def _init(self, conf):
        self.semi_dense = conf.pop('semi_dense')
        self.min_cossim = conf.pop('min_cossim')
        self.model = torch.hub.load(
            'verlab/accelerated_features', 
            'XFeat',
            pretrained = True, 
            top_k=conf["top_k"]
        ).eval()
        

    def _forward(self, data):

        B = len(data["image0"])

        if self.semi_dense:
            # Match batches of pairs
            idxs_list = self.model.batch_match(data["descriptors0"], data["descriptors1"], min_cossim=self.min_cossim)

            #Refine coarse matches
            #this part is harder to batch, currently iterate
            matches = []
            data0 = {
                "keypoints": data["keypoints0"],
                "descriptors": data["descriptors0"],
                "scales": data["scales0"]
            }
            data1 = {
                "keypoints": data["keypoints1"],
                "descriptors": data["descriptors1"],
                "scales": data["scales1"]
            }

            matches, matches0, matches1 = [], [], []
            for k in range(B):
                m = self.model.refine_matches(data0, data1, matches=idxs_list, batch_idx=k)
                m0 = m[:, :2]
                m1 = m[:, 2:]
                valid = m0 > -1
                m_indices_0 = torch.where(valid)[0]
                m_indices_1 = m0[valid]
                matches.append(torch.stack([m_indices_0, m_indices_1], -1))
                matches0.append(m0)
                matches1.append(m1)

            preds = {
                "matches": matches,
                "matches0": matches0,
                "matches1": matches1
            }

        else:
            idxs0, idxs1 = self.match(data["descriptors0"], data["descriptors1"], min_cossim=self.min_cossim)

            preds = {
                "matches": (data["keypoints0"][idxs0].cpu().numpy(), data["keypoints1"][idxs1].cpu().numpy())
            }

        return preds
