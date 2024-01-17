import cv2 as cv
import numpy as np
import math
import src.models.license_plate_recognition.tools.predict_system as predict_sys
from PIL import Image, ImageFont, ImageDraw
import re
import random

rng = np.random.default_rng(3)
PALLETE = rng.uniform(0, 255, size=(81, 3))

SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
            [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
            [3, 5], [4, 6]]

def draw_keypoints(image, keypoints, color, kpt_score_threshold=0.3, radius=4, thickness=3, skeleton=SKELETON):
    img_h, img_w, _ = image.shape
    kpts = np.array(keypoints, copy=False)
    for kpt in kpts:
        x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
        if kpt_score < kpt_score_threshold:
            continue
        cv.circle(image, (int(x_coord), int(y_coord)), radius,
                               color, -1)

    for sk in skeleton:
        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

        if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                    or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                    or pos2[1] <= 0 or pos2[1] >= img_h
                    or kpts[sk[0], 2] < kpt_score_threshold
                    or kpts[sk[1], 2] < kpt_score_threshold):
            continue
        cv.line(image, pos1, pos2, color, thickness=thickness)
    return image


def draw_results(image, model_results,ai_task="object_detection"):
        img_cpy = image.copy()
        if model_results == []:
            return img_cpy
        height, width, _ = img_cpy.shape
        text_system = predict_sys.TextSystem()
        if model_results[0]["segmentation"].size > 0:
            mask_alpha = 0.5
            for obj in model_results:
                id = int(obj["id"])
                color = PALLETE[id%PALLETE.shape[0]]
                x0 = round(obj["bbox"][0])
                y0 = round(obj["bbox"][1])
                x1 = round(obj["bbox"][2])
                y1 = round(obj["bbox"][3])
                mask_maps = obj["segmentation"]
                crop_mask = mask_maps[y0:y1, x0:x1, np.newaxis]
                crop_mask_img = img_cpy[y0:y1, x0:x1]
                crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
                img_cpy[y0:y1, x0:x1] = crop_mask_img
            img_cpy = cv.addWeighted(img_cpy, mask_alpha, image, 1 - mask_alpha, 0)

        for obj in model_results:
            x0 = round(obj["bbox"][0])
            y0 = round(obj["bbox"][1])
            x1 = round(obj["bbox"][2])
            y1 = round(obj["bbox"][3])
            id = int(obj["id"])
            class_name = obj["class"]
            confi = float(obj["confidence"])
            color = PALLETE[id%PALLETE.shape[0]]

            if obj["keypoints"].size > 0:
                img_cpy = draw_keypoints(img_cpy, obj["keypoints"], color)

               # OCR: Extract text from the region within the bounding box
            x0, y0, x1, y1 = [round(coord) for coord in obj["bbox"]]
            crop_img = img_cpy[y0:y1, x0:x1]
            if ai_task == "object_detection":
                _, ocr_text = text_system(crop_img)
                if ocr_text is not None:
                        text = '%d-%s' % (obj["id"], ocr_text[0][0])
                    # text = '%d-%s'%(id,class_name)


                        if ocr_text is not None:
                            txt_color_light = (255, 255, 1)
                            txt_color_dark = (0, 0, 0)
                            font = cv.FONT_HERSHEY_SIMPLEX
                            FONT_SCALE = 1e-3
                            THICKNESS_SCALE = 6e-4
                            font_scale = min(width, height) * FONT_SCALE
                            if font_scale <= 0.4:
                                font_scale = 0.41
                            elif font_scale > 2:
                                font_scale = 2.0
                            thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
                            txt_size = cv.getTextSize(text, font, 0.4, 1)[0]
                            cv.rectangle(img_cpy, (x0, y0), (x1, y1), color, int(thickness*5*font_scale))
                            cv.rectangle(
                            img_cpy,
                            (x0, y0 + 1),
                            (x0 + txt_size[0] + 1, y0 + int(thickness*5*font_scale)),
                            color,
                            -1)
                            text_sum = 0
                            for text, score in ocr_text[::-1]:
                                text = strip_chinese(text)
                                text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
                                text_w, text_h = text_size
                                text_sum+=text_h
                                img_cpy = draw_text(img_cpy, text,
                                                pos=(x0, y0+int(text_sum)),
                                                font=cv.FONT_HERSHEY_PLAIN,
                                                font_scale=4,
                                                text_color=(0, 0, 0),
                                                font_thickness=2,
                                                text_color_bg=(0, 255, 0))

            else:

                            text = '%d-%s'%(id,class_name)
                            txt_color_light = (255, 255, 255)
                            txt_color_dark = (0, 0, 0)
                            font = cv.FONT_HERSHEY_SIMPLEX
                            FONT_SCALE = 1e-3
                            THICKNESS_SCALE = 6e-4
                            font_scale = min(width, height) * FONT_SCALE
                            if font_scale <= 0.4:
                                font_scale = 0.41
                            elif font_scale > 2:
                                font_scale = 2.0
                            thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
                            txt_size = cv.getTextSize(text, font, 0.4, 1)[0]
                            cv.rectangle(img_cpy, (x0, y0), (x1, y1), color, int(thickness*5*font_scale))
                            cv.rectangle(
                                img_cpy,
                                (x0, y0 + 1),
                                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                                color,
                                -1)
                            cv.putText(img_cpy, text, (x0, y0 + txt_size[1]), font, font_scale, txt_color_dark, thickness=thickness+1)
                            cv.putText(img_cpy, text, (x0, y0 + txt_size[1]), font, font_scale, txt_color_light, thickness=thickness)
        return img_cpy

def draw_text(img, text,
          pos=(0, 0),
          font=cv.FONT_HERSHEY_PLAIN,
          font_scale=3,
          text_color=(0, 0, 0),
          font_thickness=2,
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(img, pos, (int(x + 1.1*text_w), y + 2*text_h), text_color_bg, -1)
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