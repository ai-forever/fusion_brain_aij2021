import re

import cv2
import numpy as np


def resize_if_need(image, max_h, max_w):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    coef = 1 if img_h <= max_h and img_w <= max_w else max(img_h / max_h, img_w / max_w)
    h = int(img_h / coef)
    w = int(img_w / coef)
    img = cv2.resize(img, (w, h))
    return img, coef


def make_img_padding(image, max_h, max_w):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    bg = np.zeros((max_h, max_w, img_c), dtype=np.uint8)
    x1 = 0
    y1 = (max_h - img_h) // 2
    x2 = x1 + img_w
    y2 = y1 + img_h
    bg[y1:y2, x1:x2, :] = img.copy()
    return bg


def simple_detect_lang(text):
    if len(set('абвгдежзийклмнопрстуфхцчшщъыьэюяё').intersection(text.lower())) > 0:
        return 'ru'
    if len(set('abcdefghijklmnopqrstuvwxyz').intersection(text.lower())) > 0:
        return 'en'
    return 'other'


class CTCLabeling:

    def __init__(self, chars, blank='ß'):
        self.blank = blank
        self.chars = [self.blank] + sorted(list(chars))
        self.char2ind = {c: i for i, c in enumerate(self.chars)}

    def encode(self, text):
        text = self.preprocess(text)
        return [self.char2ind[char] for char in text]

    def decode(self, indexes):
        chars = []
        for i, index in enumerate(indexes):
            if index == self.padding_value:
                continue
            if i == 0:
                chars.append(self.chars[index])
                continue
            if indexes[i - 1] != index:
                chars.append(self.chars[index])
                continue
        text = ''.join(chars).strip()
        text = self.postprocess(text)
        return text

    def preprocess(self, text):
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def postprocess(self, text):
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @property
    def padding_value(self):
        return self.char2ind[self.blank]

    def __len__(self):
        return len(self.chars)
