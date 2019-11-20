import pyautogui
import cv2
import numpy as np
import util.face_processing as fp

#Loading values min and ptp from file - required to normalize
#Initializing openvino model   
try:
    (m, p) = np.load("model/norm.npy", allow_pickle=True)
    fp.init(cursor_mode=True, norm_min=m, norm_ptp=p)
except:
    print('Failed to load or initialize model')

video = cv2.VideoCapture(0)

#Runtime variables used for averaging results
avx = 0
avy = 0
rou = 0
stop = False
#Program loop
while video.isOpened():
    if stop == False:
            
        ret, frame = video.read()
        if not ret:
            break
        #Passing frame into processing
        faces = fp.process(frame)

        for face in faces:
            face.draw_bbox(frame)
            face.draw_pts(frame)
            face.draw_gaze(frame)
        #Frame display
        cv2.imshow('frame', frame)

        # TODO: Select the face with the highest conf instead?
        if len(faces) is 0:
            continue
        try:
            tab = faces[0].cursor[0]
        except: 
            print('Face processing fail')

        #Screen point x,y values
        gx = tab[0]
        gy = tab[1]    

        #Averaging 10 results, displaying every 10th
        if(rou == 9):
            gx = avx/10
            gy = avy/10

            #print('x: ', gx, ' y: ', gy)
        
            if(gx < 1 and gx > 0 and gy < 1 and gy > 0):
                #if(gx<0.25):
                #    print('LEFT')
                #    pyautogui.moveTo(1920 * 0.25, 1080 * 0.5)
                #elif(gx>0.75):
                #    print('RIGHT')
                #    pyautogui.moveTo(1920 * 0.75, 1080 * 0.5)
                if(gy > 0.76):
                    print('DOWN')
                    pyautogui.moveTo(1920 * 0.5, 1080 * 0.70)
                    pyautogui.scroll(-200)

                elif(gy < 0.20):
                    print('UP')
                    pyautogui.moveTo(1920 * 0.5, 1080 * 0.25)
                    pyautogui.scroll(200)
                else:
                    print('MIDDLE')    
                    pyautogui.moveTo(1920 * 0.5, 1080 * 0.5)
            else:
                print('OUT OF SCREEN')

            rou = 0
            avx = 0
            avy = 0
        else:
            avx += gx
            avy += gy
            rou += 1



    #Input controller SPACE to exit
    #EXIT: '\'
    #PAUSE/RESUME: '|'
    key = cv2.waitKey(1)
    if key == ord(']'):
        video.release()
        break
    if key == ord('['):
        if stop == False:
            print("PAUSED")
            stop = True
        else:
            print("RESUMED")
            stop = False    
            

video.release()
cv2.waitKey(0)