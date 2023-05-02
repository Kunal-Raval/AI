import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# import pythondb
# import datetime
import pytz
import pyautogui
# import speech_recognition as sr
# import pyttsx3
# from gtts import gTTS
import os
import sys
from playsound import playsound
# playsound('audio.mp3')

arg_count = len(sys.argv)
if arg_count > 1:
    print(type(sys.argv[0]))
    threshold_angle = sys.argv[0]
    threshold_angle = int(threshold_angle)
else:
    threshold_angle = 25


mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def draw_graph(fl ):
    f = open(fl, "r")
    new_list= f.read().split(',')
    progress_lst = []
    for i in new_list:
        progress_lst.append(float(i))
    

    # print(progress_lst)
    xpoints = np.arange(1, len(progress_lst)+1)
    ypoints = np.arange(0,100,10)
    # plt.xticks(np.arange(min(xpoints), max(xpoints)+1, 2))
    # plt.yticks(np.arange(min(progress_lst), max(progress_lst)+1, 4))
    plt.xlabel('Days')
    plt.ylabel('Average Accuracy')
    plt.title('LinePlot')
    plt.plot(xpoints,progress_lst, marker = 'o')
    plt.show()



# draw_graph('accuracy.txt')


def progress_bar(right,wrong):
    
    return (right/(right+wrong))*100
    # pass
    

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
    
        
    return angle 


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#for counting reps
count = 0
green = 0
red = 0
flag =1
talk_flag = 0
_, frame = cap.read()
h, w, c = frame.shape
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) 
img_flg = True
# threshold_angle = 50

no_sets = 3
no_reps = 15

curr_set = 1
#for rendering
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        h, w, c = frame.shape
        # print(h,w,c)

     
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False# for saving memory

        results = pose.process(image)#render
        results_hands = hands.process(image)

        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if 1==1:
            
            try:
                landmarks = results.pose_landmarks.landmark
                # print(landmarks)
            
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                shoulder1 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                elbow1 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]


                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                wrist1 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                hip1 = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            
                angle = calculate_angle(shoulder, elbow, wrist)
                angle1 = calculate_angle(elbow, shoulder, hip)

                r_angle = calculate_angle(shoulder1, elbow1, wrist1)
                r_angle1 = calculate_angle(elbow1, shoulder1, hip1)

                cv2.rectangle(image,(0,0),(150,50),(0,256,0),-1)
                
                cv2.putText(image, str(count), 
                                    (15,25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
                                            )
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                height_d = 640
                height_u = 540

                width_r = 480
                width_l = 0


                if(angle1 > threshold_angle):  
                    # x_val_ = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
                    # y_val_ = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                    # x, y = int(x_val_ * w), int(y_val_ * h)
                    if talk_flag == 0:
                        playsound('beep-02.mp3',False)
                        talk_flag = 1

                    # cv2.rectangle(frame, (x-50, y-50),
                    #               (x+50, y+50), (0, 0, 255), 2)

                    cv2.rectangle(image,(0,0),(640,50),(0,0,0),-1)

                    x_val = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
                    y_val = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                    x, y = int(x_val * w), int(y_val * h)
                    # print(x,y)
                    # cv2.rectangle(frame, (x-50, y-50),
                    #               (x+50, y+50), (0, 0, 255), 2)
                    cv2.putText(image, ".", 
                                    (x,y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0, 256), 40, cv2.LINE_AA
                    )
                    if img_flg:
                        print("yes")
                        img_flg = False
                       
                        # myScreenshot = pyautogui.screenshot()
                        # if not os.path.exists('wrong'):
                        #     os.makedirs('wrong')
                        # myScreenshot.save(f'wrong/{curr_set}({count})_bicep_curl.png')

                        imagee = pyautogui.screenshot()

                        imagee = cv2.cvtColor(np.array(image),
                                            cv2.COLOR_RGB2BGR)
                        
                        # writing it to the disk using opencv
                        if not os.path.exists('wrong'):
                            os.makedirs('wrong')
                        now = datetime.now()
                        # now_time = datetime.time(pytz.timezone('India/Kolkata'))
                        # print(now_time.strftime("%H:%M:%S"))
                        dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
                        print(dt_string)
                        print("EHureka")
                        cv2.imwrite(f'wrong/{curr_set}_{count+1}_bicep_curl.png', image)
   
                  
                      
                

                    cv2.putText(image,f'current angle between your elbow and waist is {round(angle1,2)} ' ,(15,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (256, 256, 256), 2, cv2.LINE_AA)
                    cv2.putText(image,f'it should be decreased by {round(angle1-threshold_angle,2)} ' ,(15,45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (256, 256, 256), 2, cv2.LINE_AA)
                
                if(r_angle1 > threshold_angle):  
                    # x_val_ = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
                    # y_val_ = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                    # x, y = int(x_val_ * w), int(y_val_ * h)
                    if talk_flag == 0:
                        playsound('beep-02.mp3',False)
                        talk_flag = 1

                    # cv2.rectangle(frame, (x-50, y-50),
                    #               (x+50, y+50), (0, 0, 255), 2)

                    cv2.rectangle(image,(0,0),(640,50),(0,0,0),-1)

                    x_val = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
                    y_val = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                    x, y = int(x_val * w), int(y_val * h)
                    # print(x,y)
                    # cv2.rectangle(frame, (x-50, y-50),
                    #               (x+50, y+50), (0, 0, 255), 2)
                    cv2.putText(image, ".", 
                                    (x,y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0, 256), 40, cv2.LINE_AA
                    )
                    if img_flg:
                        print("yes")
                        img_flg = False
                       
                        # myScreenshot = pyautogui.screenshot()
                        # if not os.path.exists('wrong'):
                        #     os.makedirs('wrong')
                        # myScreenshot.save(f'wrong/{curr_set}({count})_bicep_curl.png')

                        imagee = pyautogui.screenshot()

                        imagee = cv2.cvtColor(np.array(image),
                                            cv2.COLOR_RGB2BGR)
                        
                        # writing it to the disk using opencv
                        if not os.path.exists('wrong'):
                            os.makedirs('wrong')
                        now = datetime.now()
                        # now_time = datetime.time(pytz.timezone('India/Kolkata'))
                        # print(now_time.strftime("%H:%M:%S"))
                        dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
                        print(dt_string)
                        print("EHureka")
                        cv2.imwrite(f'wrong/{curr_set}_{count+1}_bicep_curl.png', image)

                    cv2.putText(image,f'current angle between your elbow and waist is {round(r_angle1,2)} ' ,(15,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (256, 256, 256), 2, cv2.LINE_AA)
                    cv2.putText(image,f'it should be decreased by {round(r_angle1-threshold_angle,2)} ' ,(15,45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (256, 256, 256), 2, cv2.LINE_AA)
        
                if((angle<=55 and flag == 0) and (r_angle<=55 and flag ==0)):
                    count +=1
                    if(count % 15 == 0):
                        curr_set+=1
                    if(curr_set == 4):
                        break
                    flag = 1
                    if(angle1 < threshold_angle and r_angle < threshold_angle):
                        green += 1
                        img_flg = True  
                    else:
                        red += 1
                      

                    cv2.putText(image, str(count), 
                                    (15,25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
                                            )
                if ((angle>55 and flag ==1) and (r_angle>55 and flag == 1)):
                    flag = 0
                    talk_flag = 0

                    
    
                        
            except:
                pass
            
        
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )                 
        
        # cv2.imshow('MediaPipe Feed', cv2.flip(image, 1))
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            # markAttendanceangle(angle)
            #   (green,red)
            f = open("accuracy.txt", "a")
            f.write(',')
            f.write(str(progress_bar(green,red)))
            f.close()

            print(progress_bar(green,red))
            break
            
        
    cap.release()
    cv2.destroyAllWindows()