#!/usr/bin/env 3.9
import cv2
import os.path
import shutil
import datetime
import random
import sys
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import ImageHash
import CheckBlurness
import Enhancement

countOriginal = 1
countSharpen = 1
countFinal = 1

# parser = argparse.ArgumentParser(description='Smart Camera Command Prompt')
# parser.add_argument("-vf", "--videofile", required=True, type=str, help="This is your video file to process")
# parser.add_argument("-st", "--starttime", default=0, type=int, help="Start time of video file to process")
# parser.add_argument("-et", "--endtime", required=True, type=int, help="End time of video file to process")
# parser.add_argument("-sr", "--superres", default=1, type=int, help="Super resolution model to upscale keyframes")
# parser.add_argument("-he", "--histequal", default=0, type=int, help="Automatically adjust contrast of keyframes")
# parser.add_argument("-ae", "--autoenhance", default=1, type=int, help="Auto enhancement for keyframes")
# args = parser.parse_args()
# video_file = args.videofile
# start_time = args.starttime
# end_time = args.endtime
# superResponse = args.superres
# histResponse = args.histequal
# autoResponse = args.autoenhance

video_file = sys.argv[1]
start_time = int(sys.argv[2])
end_time = int(sys.argv[3])
superResponse = int(sys.argv[4])
histResponse = int(sys.argv[5])
autoResponse = int(sys.argv[6])

if start_time < 0:
    start_time = 0

if superResponse <= 0:
    superResponse = 1

if autoResponse <= 0:
    autoResponse = 1

if histResponse < 0:
    histResponse = 0

result_path = video_file + "_Results"  # Folder to store results
removed_path = video_file + "_Removed"
if os.path.isdir(result_path):
    shutil.rmtree(result_path)
    os.mkdir(result_path)
else:
    os.mkdir(result_path)

if os.path.isdir(removed_path):
    shutil.rmtree(removed_path)
    os.mkdir(removed_path)
else:
    os.mkdir(removed_path)


# Video information
def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


modified_video = video_file + "_modified.mp4"
video_time = str(datetime.timedelta(seconds=VideoFileClip(video_file).duration))
print("Video length: " + video_time)
video_seconds = get_sec(video_time)
if start_time >= 0 and 0 < end_time <= video_seconds:
    ffmpeg_extract_subclip(video_file, start_time, end_time, targetname=modified_video)
else:
    print("[-] Invalid input for clipping video duration")
    exit()

# Super resolution pre-trained model response
superCounter = 1
if superResponse == 1:
    superCounter = 1
elif superResponse == 2:
    superCounter = 2
elif superResponse == 3:
    superCounter = 3
else:
    print("[-] Invalid input for super resolution model")
    exit()

# Histogram Equalizer response
histCounter = 0
if histResponse == 1:
    histCounter = 1
elif histResponse == 0:
    histCounter = 0
else:
    print("[-] Invalid input for histogram equalizer")
    exit()

# Auto Enhancement response
autoCounter = 1
if autoResponse == 1:
    autoCounter = 1
elif autoResponse == 0:
    autoCounter = 0
else:
    print("[-] Invalid input for auto enhancement")
    exit()

# ================
# Motion Detection
# ================
# Capture frames from modified (clipped) video
print("[+] Running...")
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
        ImageHash.compare_images(video_file + "_Results/Final 8.jpeg", result_path, removed_path)
        keyframes_lists = os.listdir(video_file + "_Results")
        # print(keyframes_lists)
        try:
            ImageHash.compare_images(
                video_file + "_Results/" + keyframes_lists[random.randint(0, len(keyframes_lists))], result_path,
                removed_path)
        except:
            pass
        keyframes_lists = os.listdir(video_file + "_Results")
        print("Images Captured: " + str(countOriginal))
        print("Images Filtered: " + str(countFinal))
        print("Images Shortlisted: " + str(len(keyframes_lists)))
        print("\n[+] Process finished!")
        # Call mega detector function here
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
            cv2.imwrite(os.path.join(result_path, "Final " + str(countFinal) + ".jpeg"), keyframes,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            countFinal += 1
        else:
            continue

    # cv2.imshow("Output", img1)
    img1 = img2
    success, img2 = cap.read()
    # cv2.waitKey(1)

cv2.destroyAllWindows()
cap.release()
