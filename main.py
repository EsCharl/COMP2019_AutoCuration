import cv2
import os.path
import shutil
import datetime
import random
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import ImageHash
import CheckBlurness
import Enhancement

path = "Results"  # Folder to store results
video_file = "Dataset/high.mp4"
modified_video = "modified.mp4"
countOriginal = 1
countSharpen = 1
countFinal = 1

if os.path.isdir("Results"):
    shutil.rmtree("Results")
    os.mkdir("Results")
else:
    os.mkdir("Results")

if os.path.isdir("Removed"):
    shutil.rmtree("Removed")
    os.mkdir("Removed")
else:
    os.mkdir("Removed")


def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


# Video time information
video_time = str(datetime.timedelta(seconds=VideoFileClip(video_file).duration))
print("Video length: " + video_time)
video_seconds = get_sec(video_time)

# Take user input for video clipping
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

# Super resolution pre-trained model response
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

# ================
# Motion Detection
# ================
# Capture frames from modified (clipped) video
cap = cv2.VideoCapture(modified_video)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
success, img1 = cap.read()
averageValue1 = np.float32(img1)

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


while cap.isOpened():
    success, img2 = cap.read()
    if success is False:
        ImageHash.compare_images("Results/Final 8.jpeg")
        keyframes_lists = os.listdir("Results")
        # print(keyframes_lists)
        ImageHash.compare_images("Results/" + keyframes_lists[random.randint(0, len(keyframes_lists))])
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
        countOriginal += 1

        # Save specific dimension of bbox only
        if imgCaptured.shape[1] >= 300 and imgCaptured.shape[0] >= 300:
            continue

        checkBlur = CheckBlurness.check_blur(imgCaptured, height)
        if checkBlur is not None:
            countSharpen += 1
        else:
            continue

        upScale = Enhancement.upscale_image(checkBlur, superCounter)
        if histCounter == 0 and autoCounter == 0:
            keyframes = CheckBlurness.checkBlur_final(upScale, height)
        elif histCounter == 1 and autoCounter == 0:
            equalize = Enhancement.histEqualize(upScale)
            keyframes = CheckBlurness.checkBlur_final(equalize, height)
        elif histCounter == 0 and autoCounter == 1:
            auto = Enhancement.enhance_brightness_and_contrast(upScale)
            keyframes = CheckBlurness.checkBlur_final(auto, height)
        elif histCounter == 1 and autoCounter == 1:
            equalize = Enhancement.histEqualize(upScale)
            auto = Enhancement.enhance_brightness_and_contrast(equalize)
            keyframes = CheckBlurness.checkBlur_final(auto, height)
        else:
            continue

        if keyframes is not None:
            cv2.imwrite(os.path.join(path, "Final " + str(countFinal) + ".jpeg"), keyframes,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            countFinal += 1
        else:
            continue

    cv2.imshow("Output", img1)
    img1 = img2
    success, img2 = cap.read()
    cv2.waitKey(1)

    # print("Images Captured: " + str(countOriginal))
    # print("Images Filtered: " + str(countSharpen))
    # print("Images Shortlisted: " + str(countFinal))

cv2.destroyAllWindows()
cap.release()
