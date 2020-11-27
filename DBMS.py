# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 02:32:00 2020

@author: cttc
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 15:52:30 2020

@author: cttc
"""
import sqlite3 as sql
import cv2
import numpy as np
from numpy import asarray
import io
import urllib
from keras_vggface import VGGFace
from keras_vggface.utils import preprocess_input

model = VGGFace(model='resnet50',
                include_top=False,
                input_shape=(224,224,3),pooling='avg')

face_data="haarcascade_frontalface_alt.xml"

cascade=cv2.CascadeClassifier(face_data)


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sql.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sql.register_adapter(np.ndarray, adapt_array)
sql.register_converter("array", convert_array)


class Student:
    
    def preprocess(self,img):
        """helper function to preprocess image"""
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(224,224))
        img = img.reshape(1,224,224,3)
        img = asarray(img,np.float32)
        img = preprocess_input(img,version=2)
        return img        
    
    def register(self,URL,Sid,Sname,SclassId,SphoneNo,Spassword):
        
        while True:
            imgreshp=urllib.request.urlopen(URL)
            imgarray=np.array(bytearray(imgreshp.read()),dtype=np.uint8)
    
            frame=cv2.imdecode(imgarray,-1)
    
            frame=cv2.resize(frame,(680,420))
    
            faces=cascade.detectMultiScale(frame)
    
            
            for x,y,w,h in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),
                              (0,0,255),2)
                cv2.putText(frame,'press q to capture',(10,470),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                face_img = frame[y:y+h,x:x+w]
            cv2.imshow('cam',frame)
            
            if cv2.waitKey(1)==ord('q'):
                break
        cv2.destroyAllWindows()
        
        emb = model.predict(self.preprocess(face_img))
        
        curr=sql.connect("attendenceSym.db",detect_types=sql.PARSE_DECLTYPES)
        query="""INSERT INTO student_details(Sid,Sname,
                                            SclassId,
                                            SphoneNo,
                                            Spassword) values(?,?,?,?,?)"""
        curr.execute(query,(Sid,Sname,SclassId,SphoneNo,Spassword))
        
        query = """INSERT INTO student_face_data(Sid,arr) VALUES(?,?)"""
        
        curr.execute(query,(Sid,emb))
        
        
        curr.commit()
        curr.close()                                       












