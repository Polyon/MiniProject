from tkinter import *
import tkinter.messagebox
import cv2 as cv
import numpy as np
import dlib
from imutils import face_utils
import beepy
from PIL import Image, ImageTk


root= Tk()
root.geometry("650x650")
root.title("Drowsyness Detection App")
frameApp= Frame(root, relief= RIDGE, borderwidth=5, background="black")
frameApp.pack(fill= BOTH, expand= 1)
fileName= PhotoImage(file= "DDDA1.png")
background= Label(root, image= fileName)
background.pack(side= TOP, fill= BOTH)
label= Label(frameApp, text="Drowsyness Detection Application", bg="dark blue", font=("Times 30 bold italic"), fg="white")
label.pack(side= TOP, fill= BOTH)


def userHelp():
    help(cv)

def appDetails():
    tkinter.messagebox.showinfo("About", "~ Drowsyness Detection App v1.0\n ~ Made by: Polyon Mondal.\n ~ Ingrated software:\n\t1. Python v3.9\n\t2. OpenCV v4.5\n\t3. Tkinter v4.2\n\t4. Numpy")

def aboutUs():
    tkinter.messagebox.showinfo("About Us", "\tI'm Polyon Mondal. Currently I'm pursuing B.Tech on Electronics & Communication Engineering.\n\tI made this application for help people how doing a task at long time for stay awake. Besically this application detect your face and calculate your drowsyness label. When it's reach the critical label then play a bazzer to stay you awake.")

def exit():
    root.destroy()

menu= Menu(root)
root.config(menu= menu)
subMenu1= Menu(menu, tearoff=0)
menu.add_cascade(label= 'Home', menu= subMenu1)
subMenu1.add_command(label= 'Exit', command= exit)
subMenu2= Menu(menu, tearoff= 0)
menu.add_cascade(label= 'Tools', menu= subMenu2)
subMenu2.add_command(label= 'About Us', command= aboutUs)
subMenu2.add_command(label= 'Details', command= appDetails)
subMenu2.add_separator()
subMenu2.add_command(label= 'Help', command= userHelp)


PhotoFrame= LabelFrame(frameApp, bg= "black")
PhotoFrame.config(height= 450)
PhotoFrame.pack()

def DrowsynessDetect():
    FrameLabel= Label(PhotoFrame, bg= "black")
    FrameLabel.pack()

    cap= cv.VideoCapture(0)

    detector= dlib.get_frontal_face_detector()
    predictor= dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    sleep= 0
    drowsy= 0
    active= 0
    status= ''
    color= (0, 0, 0)

    def compute(ptA, ptB):
        dist= np.linalg.norm(ptA-ptB)
        return dist

    def blinked(a, b, c, d, e, f):
        up= compute(b, d)+compute(c, e)
        down= compute(a, f)
        ratio= up/(2.0*down)

        if(ratio>= 0.25):
            return 2
        elif(ratio> 0.20 and ratio< 0.25):
            return 1
        else:
            return 0

    while True:
        frame= cap.read()[1]
        gray= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces= detector(gray)

        for face in faces:
            x1= face.left()
            y1= face.top()
            x2= face.right()
            y2= face.bottom()

            face_frame= frame.copy()
            cv.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks= predictor(gray, face)
            landmarks= face_utils.shape_to_np(landmarks)

            left_blink= blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink= blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if(left_blink==0 or right_blink==0):
                sleep += 1
                drowsy= 0
                active= 0
                if(sleep>6):
                    status= "SLEEPING!!"
                    beepy.beep(sound= 3)
                    color= (255, 0, 0)
            
            elif(left_blink==1 or right_blink==1):
                sleep= 0
                active= 0
                drowsy += 1
                if(drowsy> 6):
                    status= "Drowsy!"
                    beepy.beep(sound= 1)
                    color= (0, 0, 255)
            
            else:
                drowsy= 0
                sleep= 0
                active += 1
                if(active> 6):
                    status= "Active :)"
                    color= (0, 255, 0)
            
            cv.putText(face_frame, status, (100, 100), cv.FONT_HERSHEY_COMPLEX, 1.2, color, 3)

            for n in range(0, 68):
                (x, y)= landmarks[n]
                cv.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
            img= face_frame
            img= ImageTk.PhotoImage(Image.fromarray(img))
            FrameLabel['image']= img
            frameApp.update()
            
    
def closeDetector():
    frameApp.winfo_children()[1].winfo_children()[0].destroy()
    cv.destroyAllWindows()


btn= Button(root, padx= 5, pady= 5, bg="dark green", fg="white", relief= GROOVE, command= DrowsynessDetect, text="Open Detector", font=("Arial 15 bold"))
btn.place(relx = 0.1, rely = 0.9, anchor = SW)

btn1= Button(root, padx= 5, pady= 5, bg="black", fg="white", relief= GROOVE, command= closeDetector, text="Close Detector", font=("Arial 15 bold"))
btn1.place(relx = 0.5, rely = 0.9, anchor = S)

btn2= Button(root, padx= 50, pady= 5, bg="dark red", fg="white", relief= GROOVE, command= exit, text="Exit", font=("Arial 15 bold"))
btn2.place(relx = 0.9, rely = 0.9, anchor = SE)

root.mainloop()