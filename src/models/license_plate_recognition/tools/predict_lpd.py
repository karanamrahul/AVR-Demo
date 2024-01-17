import sys
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))


import onnxruntime
from lpd.utils import im2single
from lpd.infer_utils import detect_lp_onnx

class LisencePlateDetector():
    def __init__(self) -> None:
        self.lp_threshold = 0.3
        self.lpd_onnx_path = '/home/raps/altumint_dev/Altumint_Demo_Rahul/src/models/license_plate_recognition/models/lpd.onnx'
        self.sess = onnxruntime.InferenceSession(self.lpd_onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def __call__(self, img):
        # ori_im = img.copy()
        ratio = float(max(img.shape[:2]))/min(img.shape[:2]) # ratio is used for height-width ratio and helps in resizing the image
        side = int(ratio*288.)
        bound_dim = min(side + (side % (2**4)), 608) # ca

        bbox,Llp, LlpImgs, t = detect_lp_onnx(self.sess, im2single(
            img),bound_dim, 2**4, 300, self.lp_threshold)

        return bbox,Llp, LlpImgs, t # bbox- bounding box, Llp- License Plate, LlpImgs- License Plate Images, t- time taken for detection
