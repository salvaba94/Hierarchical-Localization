from types import SimpleNamespace

from ..utils.base_model import BaseModel

from lightglue import ALIKED as ALIKED_


class ALIKED(BaseModel):
    default_conf = {
        "model_name": "aliked-n16",
        "max_num_keypoints": -1,
        "detection_threshold": 0.2,
        "nms_radius": 2,
    }
    required_data_keys = ['image']

    def _init(self, conf):
        self.model = ALIKED_(**conf).eval()

    def _forward(self, data):
        pred = self.model(data)
        return pred
