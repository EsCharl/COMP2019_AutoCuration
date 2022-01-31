import cv2
import os
import datetime
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import ImageHash
path = "Results"  # Folder to store results


def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


# Create directory
os.mkdir("Results")
os.mkdir("Removed")

# Find video length
video_file = "Dataset/high.mp4"
clip = VideoFileClip(video_file)
video_time = str(datetime.timedelta(seconds=clip.duration))
print("Video length: " + video_time)
video_seconds = get_sec(video_time)
# print(video_seconds)

# Clip video and ask user to enter start and end time
modified_video = "modified.mp4"
start_time = int(input("Enter start time:\n>> "))
select_time = int(input("1: Whole video | 2: Select specific time\n"))
if select_time == 1:
    end_time = video_seconds
else:
    end_time = int(input("Enter end time:\n>> "))
if start_time >= 0 and 0 < end_time <= video_seconds:
    ffmpeg_extract_subclip(video_file, start_time, end_time, targetname=modified_video)
else:
    print("Invalid input")
    exit()

# Super resolution pre-trained model
superCounter = 1
print("1: ESPCN | 2:FSRCNN | 3: LapSRN")
superResponse = int(input("Which pre-trained model do you prefer?\n>> "))
if superResponse == 1:
    superCounter = 1
elif superResponse == 2:
    superCounter = 2
elif superResponse == 3:
    superCounter = 3
else:
    print("Invalid input")
    exit()


# Histogram Equalizer response
histCounter = 0
histResponse = int(input("Do you want to apply Histogram Equalizer?\n>> "))
if histResponse == 1:
    histCounter = 1
elif histResponse == 0:
    histCounter = 0
else:
    print("Invalid input")
    exit()


# Auto Enhancement response
autoCounter = 0
autoResponse = int(input("Do you want to apply Auto Enhancement?\n>> "))
if autoResponse == 1:
    autoCounter = 1
elif autoResponse == 0:
    autoCounter = 0
else:
    print("Invalid input")
    exit()


# Start modified video and capture frames
cap = cv2.VideoCapture(modified_video)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
countOriginal = 1
countSharpen = 1
countFinal = 1

# Capture initial image
success, img1 = cap.read()
averageValue1 = np.float32(img1)

# skip frames
fps = 30

if video_time >= str("0:30:00"):
    fps = 80
elif video_time >= str("0:15:00"):
    fps = 40
else:
    fps = 20


# Basic background subtraction
def background_subtraction(imgBS_one, imgBS_two):
    imgDiff = cv2.absdiff(imgBS_one, imgBS_two)
    imgGray = cv2.cvtColor(imgDiff, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
    _, imgThresh = cv2.threshold(imgBlur, 20, 255, cv2.THRESH_BINARY)
    imgDilated = cv2.dilate(imgThresh, None, iterations=3)
    imgContours_BS, _ = cv2.findContours(imgDilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return imgContours_BS


# Sharpen images and return the ones with higher value (Depends on video quality)
def check_blur(frames):
    #print("check first")
    image_sharp = cv2.detailEnhance(frames, sigma_s=3, sigma_r=0.3)
    imgGrayBlurUpdated = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2GRAY)
    valueUpdated = round(cv2.Laplacian(imgGrayBlurUpdated, cv2.CV_64F).var())
    #print(valueUpdated)
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


def checkBlur_final(frames):
    #print("check")
    imgGrayBlurUpdated = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    valueUpdated = round(cv2.Laplacian(imgGrayBlurUpdated, cv2.CV_64F).var())
    #print(valueUpdated)
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


while cap.isOpened():
    success, img2 = cap.read()
    if success is False:
        ImageHash.compare_images()
        print("\nProcess finished!")
    # Skip frames function
    cf = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, cf + fps)

    # Advance background subtraction
    cv2.accumulateWeighted(img2, averageValue1, 0.2)
    resultFrame1 = cv2.convertScaleAbs(averageValue1)
    imgContours = background_subtraction(img1, resultFrame1)

    for contour in imgContours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 16000:
            continue
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        imgCaptured = img1[y: y + h, x:x + w]

        # cv2.imwrite(os.path.join(path, "Original " + str(countOriginal) + ".jpeg"), imgCaptured)
        countOriginal += 1
        # Save specific dimension of bbox only
        if imgCaptured.shape[1] >= 300 and imgCaptured.shape[0] >= 300:
            continue

        checkBlur = check_blur(imgCaptured)
        if checkBlur is not None:
            # cv2.imwrite(os.path.join(path, "Sharpen " + str(countSharpen) + ".jpeg"), checkBlur)
            countSharpen += 1
        else:
            continue

        upScale = upscale_image(checkBlur, superCounter)
        if histCounter == 0 and autoCounter == 0:
            keyframes = checkBlur_final(upScale)
        elif histCounter == 1 and autoCounter == 0:
            equalize = histEqualize(upScale)
            keyframes = checkBlur_final(equalize)
        elif histCounter == 0 and autoCounter == 1:
            auto = enhance_brightness_and_contrast(upScale)
            keyframes = checkBlur_final(auto)
        elif histCounter == 1 and autoCounter == 1:
            equalize = histEqualize(upScale)
            auto = enhance_brightness_and_contrast(equalize)
            keyframes = checkBlur_final(auto)
        else:
            continue

        if keyframes is not None:
            cv2.imwrite(os.path.join(path, "Final " + str(countFinal) + ".jpeg"), keyframes, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            countFinal += 1
        else:
            continue

    cv2.imshow("Output", img1)
    img1 = img2
    success, img2 = cap.read()
    cv2.waitKey(1)

    print("Images Captured: " + str(countOriginal))
    print("Images Filtered: " + str(countSharpen))
    print("Images Shortlisted: " + str(countFinal))

cv2.destroyAllWindows()
cap.release()
