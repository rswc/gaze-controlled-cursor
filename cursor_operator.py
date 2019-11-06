import pyautogui
import cv2
import numpy as np
import face_processing as fp

#pyautogui.moveTo(100,150)
x,y = pyautogui.position()

print("pos: ", x, ' ', y)

# TODO: min & ptp values should be constant, or saved & loaded along with
# cursor model data. F_P needs to be initialized with the correct values
fp.init(cursor_mode=True,
        norm_min=np.array([171.0, 177.0, 217.0, 178.0, -0.6136980056762695, -0.3205522298812866,
                          -0.9682818651199341, 13268.0, -35.98328399658203, -11.750590324401855,
                          -39.597164154052734]),
        norm_ptp=np.array([237.0, 97.0, 248.0, 101.0, 1.2162402272224426, 0.49350541830062866,
                           0.24336284399032593, 22156.0, 73.28462600708008, 19.41169261932373,
                           67.85696792602539]))

video = cv2.VideoCapture(0)



avx = 0
avy = 0
rou = 0
speed = 1
prev = 'MIDDLE'

#Program loop
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    faces = fp.process(frame)

    for face in faces:
        face.draw_bbox(frame)
        face.draw_pts(frame)
        face.draw_gaze(frame)

    cv2.imshow('frame', frame)

    # TODO: Select the face with the highest conf instead?
    if len(faces) is 0:
        continue

    face = faces[0]

    #print('normalized: ', t)
    #print('pred: ', model.predict(t))
    #print('exp: ', testing_labels[600])
    tab = face.cursor[0]

    gx = tab[0]
    gy = tab[1]
    print('x = ', float(int(gx*1000)/10), '| y = ',float(int(gy*1000)/10))
    

    #UŚREDNIANIE 5 WYNIKÓW, WYŚWIETLANIE CO 5
    if(rou == 9):
        gx = avx/10
        gy = avy/10
        
        # print('x = ', float(int(gx*1000)/10), '| y = ',float(int(gy*1000)/10))
    #    if(gx < 1 and gx > 0 and gy < 1 and gy > 0):
    #        pyautogui.moveTo(1920*gx,1080*gy)
        if(gx < 1 and gx > 0 and gy < 1 and gy > 0):
            #if(gx<0.25):
            #    print('LEFT')
            #    pyautogui.moveTo(1920 * 0.25, 1080 * 0.5)
            #elif(gx>0.75):
            #    print('RIGHT')
            #    pyautogui.moveTo(1920 * 0.75, 1080 * 0.5)
            if(gy > 0.65):
                print('DOWN')
                pyautogui.moveTo(1920 * 0.5, 1080 * 0.70)
                pyautogui.scroll(-200*speed)

            elif(gy < 0.30):
                print('UP')
                pyautogui.moveTo(1920 * 0.5, 1080 * 0.25)
                pyautogui.scroll(200*speed)
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

    key = cv2.waitKey(1)
    if key == ord(' '):
        video.release()
        break
   

video.release()
cv2.waitKey(0)