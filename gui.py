from tkinter import *
import tkinter
import tkinter.font
import sys
import os

root = Tk()
root.geometry("330x600")
changefont = tkinter.font.Font(size=50)
judl = Label(root,text = "SMART DOOR LOCK")
judl.place(x =100,y = 10)
def run():
    os.system('take.py')
def run2():
    os.system('facetest.py')
btn = Button(root,text = "take picture", command=run,height=5,width=25).place(x = 80,y = 300)
btn = Button(root,text = "face detection",command=run2,height=5,width=25).place(x = 80,y = 400)
root.mainloop()