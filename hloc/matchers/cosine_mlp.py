'''
Code for loading XFeat as a global features extractor.

XFeat paper: https://arxiv.org/abs/2404.19174
'''

import torch
import torch.nn.functional as F

from ..utils.base_model import BaseModel


class CosineMLP(BaseModel):
    default_conf = {
        'top_k' : 4096,
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
        self.min_cossim = conf.pop('min_cossim')
        self.model = torch.hub.load(
            'verlab/accelerated_features', 
            'XFeat',
            pretrained = True, 
            top_k=conf["top_k"]
        ).eval()
        

    def _refine_matches(
        self,
        d0, 
        d1, 
        matches, 
        batch_idx, 
        fine_conf = 0.25
    ):
        idx0, idx1 = matches[batch_idx]
        feats1 = d0['descriptors'][batch_idx][idx0]
        feats2 = d1['descriptors'][batch_idx][idx1]
        mkpts_0 = d0['keypoints'][batch_idx][idx0]
        mkpts_1 = d1['keypoints'][batch_idx][idx1]
        sc0 = d0['scales'][batch_idx][idx0]

        #Compute fine offsets
        offsets = self.model.net.fine_matcher(torch.cat([feats1, feats2],dim=-1))
        conf = F.softmax(offsets*3, dim=-1).max(dim=-1)[0]
        offsets = self.model.subpix_softmax2d(offsets.view(-1,8,8))

        mkpts_0 += offsets* (sc0[:,None]) #*0.9 #* (sc0[:,None])

        mask_good = conf > fine_conf
        mkpts_0 = mkpts_0[mask_good]
        mkpts_1 = mkpts_1[mask_good]
        conf = conf[mask_good]

        return torch.cat([mkpts_0, mkpts_1], dim=-1), conf


    def _forward(
        self, 
        data
    ):

        B = len(data["image0"])

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

        matches0, matching_scores0 = [], []
        for k in range(B):
            m, conf = self._refine_matches(data0, data1, matches=idxs_list, batch_idx=k)
            m0 = m[:, :2]
            valid = m0 > -1
            m_indices_0 = torch.where(valid)[0]
            conf = conf[valid[:, 0]]
            matches0.append(m_indices_0)
            matching_scores0.append(conf)

        preds = {
            "matches0": matches0,
            "matching_scores0": matching_scores0,
        }

        return preds
