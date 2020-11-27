# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 01:21:14 2020

@author: cttc
"""

import sqlite3 as sql
import io
import numpy as np

con = sql.connect('Attendance_System.db')
con.close()

""" Table 1:- Student Details Creation"""

con = sql.connect('Attendance_System.db')

query = """ CREATE TABLE Student_deatils(SId TEXT PRIMARY KEY,
Sname TEXT, Sclass_Id TEXT,Sphone_no INTEGER, 
Spassword TEXT)"""

con.execute(query)
con.commit()
con.close()

""" Configuring data base 
"""
def adapt_array(arr):
   
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sql.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Converts np.array to TEXT when inserting
sql.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sql.register_converter("array", convert_array)



"""Creating Student_Face_data Table"""

con = sql.connect('Attendance_System.db',
                  detect_types = sql.PARSE_DECLTYPES)
query = """CREATE TABLE Student_face_data(SId TEXT,
arr array, FOREIGN KEY(SId) REFERENCES Student_deatils(SId)
ON DELETE CASCADE ON UPDATE CASCADE)"""


con.execute(query)
con.commit()
con.close()





