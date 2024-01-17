import os
import cv2
import time
import numpy as np
import src.models.license_plate_recognition.tools.predict_system as predict_sys
import src.models.license_plate_recognition.tools.predict_lpd as predict_lpd
from PIL import Image, ImageFont, ImageDraw
import re
import random


class ALPR():
    def __init__(self, out_dir: str) -> None:
        self.text_system = predict_sys.TextSystem() # Initialize the text recognition system
        self.lp_detector = predict_lpd.LisencePlateDetector() # Initialize the license plate detection system
        self.output_directory = out_dir

    def recognize_lp(self, img , save : bool, show : bool, f_scale : float):

        # img = cv2.imread(path)
        # Name = os.path.basename(path).split('.')[0]
        print("Image Shape",img.shape)
        bbox, _,  LlpImgs,_ = self.lp_detector(img)
        detection_results = []  # List to store results
        if bbox is not None:

            if len(bbox):
                pts = bbox[0]
                xmin = int(min(pts[0]))
                ymin = int(min(pts[1]))
                xmax = int(max(pts[0]))
                ymax = int(max(pts[1]))
                Width = int(xmax-xmin)
                cv2.rectangle(img, (xmin,ymin),(xmax,ymax), (0, 255, 0), int(2/1.5*f_scale))
                Ilp = LlpImgs[0]
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
                _, rec_res = self.text_system(Ilp*255.)
                if rec_res is not None:
                    text_sum = 0
                    for text, score in rec_res[::-1]:
                        text = strip_chinese(text)
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, f_scale, 2)
                        text_w, text_h = text_size
                        text_sum+=text_h
                        img = draw_text(img, text,
                                        pos=(xmin, ymin-int(text_sum)),
                                        font=cv2.FONT_HERSHEY_PLAIN,
                                        font_scale=f_scale,
                                        text_color=(0, 0, 0),
                                        font_thickness=2,
                                        text_color_bg=(0, 255, 0)
                                        )

                obj_dict = {
                        "id": random.randint(0, 100),
                        'class': 'license_plate',
                        'confidence': 0.5,  # Add confidence if available
                        'bbox': [xmin, ymin, Width, ymax],
                        'text': rec_res[0][0] if rec_res is not None else "",
                        'image': img,
                        "keypoints": np.array([]),
                        "segmentation": np.array([])
                    }

                detection_results.append(obj_dict)

            return detection_results

    def process_video(self,frame):
        # print("Processing video..."
        #       f"\n\tVideo path: {video_path}")
        # import os

        # if not os.path.exists(video_path):
        #     print(f"File not found: {video_path}")
        #     return
        # cap = cv2.VideoCapture(video_path)
        # if not cap.isOpened():
        #     print("Error: Could not open video.")
        #     return

        # frame_id = 0
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if not ret:
        #         break

            # Apply ALPR to the frame
            # detection_results = self.recognize_lp(frame, save=False, show=False, f_scale=1.5)
            bbox,_,LlpImgs,_ = self.lp_detector(frame)
            detection_results = []  # List to store results
            if len(bbox):
                pts = bbox[0]
                xmin = int(min(pts[0]))
                ymin = int(min(pts[1]))
                xmax = int(max(pts[0]))
                ymax = int(max(pts[1]))
                Width = int(xmax-xmin)
                cv2.rectangle(frame, (xmin,ymin),(xmax,ymax), (0, 255, 0), int(2/1.5*1))
                Ilp = LlpImgs[0]
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
                _, rec_res = self.text_system(Ilp*255.)
                if rec_res is not None:
                    text_sum = 0
                    for text, score in rec_res[::-1]:
                        text = strip_chinese(text)
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        text_w, text_h = text_size
                        text_sum+=text_h
                        frame = draw_text(frame, text,
                                        pos=(xmin, ymin-int(text_sum)),
                                        font=cv2.FONT_HERSHEY_PLAIN,
                                        font_scale=1,
                                        text_color=(0, 0, 0),
                                        font_thickness=2,
                                        text_color_bg=(0, 255, 0))

                obj_dict = {    "id": random.randint(0, 100),
                                'class': 'license_plate',
                                'confidence': 0.5,  # Add confidence if available
                                'bbox': [xmin, ymin, Width, ymax],
                                'text': rec_res[0][0] if rec_res is not None else "",
                                'image': frame,
                                "keypoints": np.array([]),
                                "segmentation": np.array([])
                            }

                detection_results.append(obj_dict)

            return detection_results




def draw_text(img, text,
          pos=(0, 0),
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=3,
          text_color=(0, 0, 0),
          font_thickness=2,
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (int(x + 1.1*text_w), y + 2*text_h), text_color_bg, -1)
    im_p = Image.fromarray(img)
    draw = ImageDraw.Draw(im_p)
    font = ImageFont.truetype("/home/raps/altumint_dev/Altumint_Demo_Rahul/src/models/license_plate_recognition/fonts/simfang.ttf",int(32*font_scale/1.5))
    draw.text((x, y ),text,text_color,font=font)
    result_o = np.array(im_p)
    # cv2.putText(img, text, (x, int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness)
    return result_o

def strip_chinese(string):
    en_list = re.findall(u'[^\u4E00-\u9FA5]', string)
    for c in string:
        if c not in en_list:
            string = string.replace(c, '')
    return string