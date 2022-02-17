import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.debug.examples.debug_mnist import tf

from Model.model import Emotion_model
from PIL import Image
emotions_names = {0: 'anger',
                  1: 'disgust',
                  2: 'fear',
                  3: 'happiness',
                  4: 'sadness',
                  5: 'surprise',
                  6: 'neutral'
                 }


def prepare_data(df):
    img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48).astype('uint8'))
    img_array = np.stack(img_array, axis=0)
    img_labels = []
    for i in df.emotion:
        img_labels.append(emotions_names[int(i)])
    return img_array, img_labels

im = Image.open("img1.jpg")
np_im = np.array(im)

model = Emotion_model.create_model("ForSwift", [])
df = model.load_data()
img, labels = prepare_data(df)


for i in range(len(img)):
    folder = labels[i]
    im = Image.fromarray(img[i], mode="L")
    im.save("Faces/{0}/{1}.jpeg".format(folder, i))
