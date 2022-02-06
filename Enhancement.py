import cv2
import numpy as np


# Upscale image 2 times
def upscale_image(img_checked, counter):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    if counter == 1:
        model = "SR_Models/ESPCN_x2.pb"
        set_model = "espcn"
    elif counter == 2:
        model = "SR_Models/FSRCNN_x2.pb"
        set_model = "fsrcnn"
    elif counter == 3:
        model = "SR_Models/LapSRN_x2.pb"
        set_model = "lapsrn"
    sr.readModel(model)
    sr.setModel(set_model, 2)
    result = sr.upsample(img_checked)
    # resized = cv2.resize(img_checked, dsize=None, fx=2, fy=2)
    return result


def histEqualize(frames):
    ycrcb = cv2.cvtColor(frames, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    equalizeImg = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return equalizeImg


def enhance_brightness_and_contrast(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(img, -1, kernel)
    return image_sharp
