import pyautogui
import cv2
import face_processing as fp


#pyautogui.moveTo(100,150)
x,y = pyautogui.position()

print("pos: ", x, ' ', y)

####################
face_avg = fp.GazeVectorAverager(10)



camera = cv2.VideoCapture(0) #create a VideoCapture object with the 'first' camera (your webcam)

while(True):
    ret, frame = camera.read()             # Capture frame by frame   

    faces = fp.process(frame)


    cv2.imshow('Press Spacebar to Exit',frame)              # Display the frame
    
    if cv2.waitKey(1) & 0xFF == ord(' '):  # Stop if spacebar is detected
        break

#camera.release()                           # Cleanup after spacebar is detected.
#cv2.destroyAllWindows()