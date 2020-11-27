# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 02:39:15 2020

@author: cttc
"""

from PyQt5 import uic,QtWidgets
from dbms import Student


def gotostudent():
    frontPage.close()
    studentpage.show()

def gotohome():
    studentpage.close()
    frontPage.show()
    
def gotofaculty():
    frontPage.close()
    facultyLoginpage.show()
    
def backhome():
    facultyLoginpage.close()
    frontPage.show()
    
def gotoadmin():
    frontPage.close()
    adminLoginpage.show()
    
def backfront():
    adminLoginpage.close()
    frontPage.show()
    
def gotodash():
    facultyLoginpage.close()
    facultyDash.show()
    
def bachome():
    facultyDash.close()
    frontPage.show()
    
def register():
    facultyDash.close()
    registerPage.show()
    
def backdash():
    registerPage.close()
    facultyDash.show()
    
def capturedata():
    Sid=registerPage.lineEdit_2.text()
    Sname=registerPage.lineEdit.text()
    SclassId=registerPage.lineEdit_3.text()
    SphoneNo=registerPage.lineEdit_2.text()
    Spassword=registerPage.lineEdit_4.text()
    URL=registerPage.lineEdit_6.text()
    
    s=Student()
    s.register(URL,Sid,Sname,SclassId,SphoneNo,Spassword)
    



app = QtWidgets.QApplication([])

frontPage = uic.loadUi('UiFiles/FrontPage.ui')
studentpage = uic.loadUi('UiFiles/student_login.ui')
facultyLoginpage=uic.loadUi('UiFiles/faculty_login.ui')
adminLoginpage=uic.loadUi('UiFiles/admin_login.ui')
facultyDash=uic.loadUi('UiFiles/faculty_dashboard.ui')
registerPage=uic.loadUi('UiFiles/student_register.ui')

frontPage.show()

frontPage.pushButton.clicked.connect(gotostudent)
studentpage.pushButton_2.clicked.connect(gotohome)
frontPage.pushButton_2.clicked.connect(gotofaculty)
facultyLoginpage.pushButton_2.clicked.connect(backhome)
frontPage.pushButton_3.clicked.connect(gotoadmin)
adminLoginpage.pushButton_2.clicked.connect(backfront)
facultyLoginpage.pushButton.clicked.connect(gotodash)
facultyDash.pushButton_4.clicked.connect(bachome)
facultyDash.pushButton_3.clicked.connect(register)
registerPage.pushButton.clicked.connect(backdash)
registerPage.pushButton_2.clicked.connect(capturedata)


app.exec()
