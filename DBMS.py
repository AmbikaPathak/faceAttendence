# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 02:32:00 2020

@author: cttc
"""

import cv2
import numpy as np
from keras.engine import Model
from keras.layers import Input
from keras_vggface import VGGFace
from keras_vggface.utils import preprocess_input
import urllib
import io
from numpy import asarray


URL = "http://192.168.42.129:8080/shot.jpg"
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

model = VGGFace(model='resnet50',
                include_top=False,
                input_shape=(224,
                             224,3),
                pooling='avg')