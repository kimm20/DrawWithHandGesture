import cv2
import mediapipe as mp
import time
import numpy as np

#------------- settings -------------#
smoothingsample = 10#max sample for smoothing effect
cap = cv2.VideoCapture(0)#cam id (left 0 if not specify)
canvassize=(800,600)
# Do not edit below if you dont know what you are doing
#------------- settings -------------#

pTime = 0
cTime = 0
penimg = cv2.resize(cv2.imread("pen.png"),(100,100))
eraserimg = cv2.resize(cv2.imread("eraser.png"),(100,100))
canvas = [255, 255, 255] * np.ones((canvassize[1], canvassize[0], 3), np.uint8)
#canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
canvas = np.uint8(np.absolute(canvas))
def calcdis(list1,list2):
    x2,y2=list2
    x1,y1=list1
    return np.round(np.sqrt((x2-x1)**2+(y2-y1)**2))
def midpoint(cod1,cod2):
    x1,y1=cod1
    x2,y2=cod2
    coord=(int((x1+x2)/2),int((y1+y2)/2))
    return coord

def merge_image(back, front, x,y):

    if back.shape[2] == 3:
        back = cv2.cvtColor(back, cv2.COLOR_BGR2BGRA)
    if front.shape[2] == 3:
        front = cv2.cvtColor(front, cv2.COLOR_BGR2BGRA)

    bh,bw = back.shape[:2]
    fh,fw = front.shape[:2]
    x1, x2 = max(x, 0), min(x+fw, bw)
    y1, y2 = max(y, 0), min(y+fh, bh)
    front_cropped = front[y1-y:y2-y, x1-x:x2-x]
    back_cropped = back[y1:y2, x1:x2]

    alpha_front = front_cropped[:,:,3:4] / 255
    alpha_back = back_cropped[:,:,3:4] / 255
    
    result = back.copy()
    #print(f'af: {alpha_front.shape}\nab: {alpha_back.shape}\nfront_cropped: {front_cropped.shape}\nback_cropped: {back_cropped.shape}')
    result[y1:y2, x1:x2, :3] = alpha_front * front_cropped[:,:,:3] + (1-alpha_front) * back_cropped[:,:,:3]
    result[y1:y2, x1:x2, 3:4] = (alpha_front + alpha_back) / (1 + alpha_front*alpha_back) * 255

    return result

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.8,static_image_mode=False,max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
lastestmove=[]
coordmap=[]
rtg=False
gtb=False
btr=False
step=5
curcolr=[255,0,0]
drawx=0
drawy=0
drawingto=penimg
while True:
    try:
        cpcanvas = canvas.copy()
        if curcolr[0]>=255:
            rtg=True
            btr=False
        elif curcolr[1]>=255:
            gtb=True
            rtg=False
        elif curcolr[2]>=255:
            btr=True
            gtb=False
        if rtg:
            curcolr[0]-=step
            curcolr[1]+=step
        elif gtb:
            curcolr[1]-=step
            curcolr[2]+=step
        elif btr:
            curcolr[2]-=step
            curcolr[0]+=step
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    coordmap.append((cx,cy))
                    cv2.circle(img, (cx, cy), 15, curcolr, cv2.FILLED)
                    #cv2.putText(img,str((cx,cy)),(cx-85,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(140, 255, 0),1,cv2.LINE_AA)
                #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            dis8t12=calcdis(coordmap[8],coordmap[12])
            dis12t16=calcdis(coordmap[12],coordmap[16])
            cv2.circle(img, coordmap[12], 20, (0,0,255), cv2.FILLED)
            if dis8t12<=50:
                cv2.circle(img, midpoint(coordmap[8],coordmap[12]), 8, (0,0,255), cv2.FILLED)
                cv2.putText(cpcanvas,"Draw mode",(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(140, 255, 0),1,cv2.LINE_AA)
                cv2.circle(canvas, (int(((1280-coordmap[12][0])/1280)*canvassize[0]),int((coordmap[12][1]/720)*canvassize[1])+100), 8, (0,0,0), cv2.FILLED)
                img = cv2.line(img, coordmap[8], coordmap[12], [0,255,0], 3)    
            elif dis12t16<=50:
                cv2.circle(img, midpoint(coordmap[16],coordmap[12]), 8, (0,0,255), cv2.FILLED)
                img = cv2.line(img, coordmap[16], coordmap[12], [0,255,0], 3)
                cv2.putText(cpcanvas,"Delete mode",(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1,cv2.LINE_AA)
                cv2.circle(canvas, (int(((1280-coordmap[12][0])/1280)*canvassize[0]),int((coordmap[12][1]/720)*canvassize[1])+100), 10, (255,255,255), cv2.FILLED)
                drawingto=eraserimg
            else:
                img = cv2.line(img, coordmap[16], coordmap[12], [0,0,255], 2)
                drawingto=penimg
                img = cv2.line(img, coordmap[8], coordmap[12], [0,0,255], 2)
                
            
            
            cv2.putText(img,str(dis12t16),midpoint(coordmap[12],coordmap[16]),cv2.FONT_HERSHEY_SIMPLEX,1,(140, 255, 0),1,cv2.LINE_AA)
            cv2.putText(img,str(dis8t12),midpoint(coordmap[8],coordmap[12]),cv2.FONT_HERSHEY_SIMPLEX,1,(140, 255, 0),1,cv2.LINE_AA)
            print(coordmap[12])
            drawx,drawy=coordmap[12]
            drawx = int(((1280-drawx)/1280) *canvassize[0])
            drawy = int((drawy/720) *canvassize[1])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.imshow("hand", cv2.resize(img,(400,300)))
        
        cpcanvas = merge_image(cpcanvas,drawingto,drawx,drawy)
        cv2.imshow("canvas",cpcanvas)
        cv2.waitKey(1)
        coordmap=[]
    except:
        pass
