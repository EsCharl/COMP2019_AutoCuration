import cv2


def check_blur(frames, height):
    image_sharp = cv2.detailEnhance(frames, sigma_s=3, sigma_r=0.3)
    imgGrayBlurUpdated = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2GRAY)
    valueUpdated = round(cv2.Laplacian(imgGrayBlurUpdated, cv2.CV_64F).var())
    if (height == 1080) and (valueUpdated > 1600):
        return image_sharp
    elif (height == 720) and (valueUpdated > 1200):
        return image_sharp
    elif (height == 480) and (valueUpdated > 800):
        return image_sharp
    elif (height == 144) and (valueUpdated > 600):
        return image_sharp
    else:
        return None


def checkBlur_final(frames, height):
    imgGrayBlurUpdated = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    valueUpdated = round(cv2.Laplacian(imgGrayBlurUpdated, cv2.CV_64F).var())
    if (height == 1080) and (valueUpdated > 160):
        return frames
    elif (height == 720) and (valueUpdated > 140):
        return frames
    elif (height == 480) and (valueUpdated > 120):
        return frames
    elif (height == 144) and (valueUpdated > 100):
        return frames
    else:
        return None

