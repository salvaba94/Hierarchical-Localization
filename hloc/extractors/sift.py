
from ..utils.base_model import BaseModel

from lightglue import SIFT as SIFT_


class SIFT(BaseModel):
    default_conf = {
        "rootsift": True,
        "nms_radius": 0,  # None to disable filtering entirely.
        "max_num_keypoints": 4096,
        "backend": "opencv",  # in {opencv, pycolmap, pycolmap_cpu, pycolmap_cuda}
        "detection_threshold": 0.0066667,  # from COLMAP
        "edge_threshold": 10,
        "first_octave": -1,  # only used by pycolmap, the default of COLMAP
        "num_octaves": 4,
    }
    required_data_keys = ['image']

    def _init(self, conf):
        self.model = SIFT_(**conf).eval()

    def _forward(self, data):
        pred = self.model(data)
        pred["descriptors"] = pred["descriptors"].transpose(-1, -2)
        pred["scores"] = pred.pop("keypoint_scores")

        return pred
