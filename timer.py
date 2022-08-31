import os
import pandas as pd
import torch
from xml_to_YOLOformat import date_completed
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_path = "./train_results/nano_train/weights/best.pt"
model = torch.hub.load('/home/ec2-user/yolov5', 'custom',path = model_path, source = "local")
model2 = torch.hub.load('/home/ec2-user/yolov5', 'custom',path = model_path, source = "local")
model2.conf = 0.002
# model.load_state_dict(torch.load(model_path)['model'].state_dict())

def get_true(f_name, df):
    n_con_name = f_name.split(".")[0]
    n_con = int(df[df["Position"] == n_con_name]["SpotCount"])
    well_A_name = n_con_name.replace(n_con_name[0],chr(ord(n_con_name[0])+1))
    well_A = int(df[df["Position"] == well_A_name]["SpotCount"])
    well_B_name = well_A_name.replace(well_A_name[0],chr(ord(well_A_name[0])+1))
    well_B = int(df[df["Position"] == well_B_name]["SpotCount"])
    p_con_name = well_B_name.replace(well_B_name[0],chr(ord(well_B_name[0])+1))
    p_con = int(df[df["Position"] == p_con_name]["SpotCount"])
    return n_con,well_A,well_B,p_con

def get_results(root, f_name):
    n_con_path = os.path.join(root,f_name)
    well_A_path = os.path.join(root,f_name.replace(f_name[0],chr(ord(f_name[0])+1)))
    well_B_path = os.path.join(root,f_name.replace(f_name[0],chr(ord(f_name[0])+2)))
    p_con_path = os.path.join(root,f_name.replace(f_name[0],chr(ord(f_name[0])+3)))
    one_to_three_imgs = [n_con_path,well_A_path,well_B_path]
    fourth_imgs = [p_con_path]
    time_p3 = time.time()
    results = model(one_to_three_imgs)
    results2 = model2(fourth_imgs)
    time_p4 = time.time()
    readtime = time_p4 - time_p3
    return results,results2,readtime


def get_pred(results,results2):
    pd,pd2 = results.pandas().xyxy, results2.pandas().xyxy
    return len(pd[0]),len(pd[1]),len(pd[2]),len(pd2[0]) #return n_con,well_A,well_B,p_con

def save_wrong_pred(results,root,f_name,true,pred):
    plate_ID = root.split("/")[-1]
    well_num = f_name.split(".")[0]
    results.save(save_dir=f'runs/wrong_pred/{true}⇒{pred}/{plate_ID}/{well_num}')

def find_csv(root,csv_root):
    folder_name = root.split("/")[-1]
    csv_path = None
    for r, d, f in os.walk(csv_root):
        for F in f:
            if F == folder_name + ".csv":
                csv_path = os.path.join(r,F)
    assert csv_path != None
    return csv_path


def detect_number(n_con,well_A,well_B,p_con):
    if n_con >10 :
        return int(3)#判定不能
    else:
        numberA = well_A - n_con
        numberB = well_B - n_con
        number = int(max(numberA,numberB))
        if number > 5:
            if number >= 8:
                return int(1)#陽性
            else:return int(2)#再検査
        else:
            if p_con < 20:
                return int(3)#判定不能
            else:
                if number <=4:
                    return int(0)#陰性
                else:return int(2)#再検査

def detect_number2(n_con,well_A,well_B,p_con):
    if n_con >10 :
        return int(3)#判定不能
    else:
        numberA = well_A - n_con
        numberB = well_B - n_con
        number = int(max(numberA,numberB))
        if number > 5:
            if number >= 8:
                return int(1)#陽性
            else:return int(2)#再検査
        else:
            if p_con < 20:
                return int(3)#判定不能
            else:
                if number <=3:
                    return int(0)#陰性
                else:return int(2)#再検査


date_completed = ["2018-04-01",
                  "2018-12-07",
                  "2019-04-08",
                  "2019-08-01"
                  ]

first_pic = ['A1.png', 'A10.png', 'A11.png', 'A12.png', 'A2.png', 'A3.png', 'A4.png', 'A5.png', 'A6.png', 'A7.png', 'A8.png', 'A9.png',
            'E1.png', 'E10.png', 'E11.png', 'E12.png', 'E2.png', 'E3.png', 'E4.png', 'E5.png', 'E6.png', 'E7.png', 'E8.png', 'E9.png']
y_pred = [];y_true = []
count = 0
time_for_sample = 0
time_for_read = 0
for date in date_completed:
    path = f"./well_png/{date}/"
    for root, dirs, files in os.walk(path):
        if dirs !=[]:
            continue
        csv_path = find_csv(root,"./export_data/")
        df = pd.read_csv(csv_path)
        for f_name in first_pic:
            if f_name not in files:
                    continue
            time_p1 = time.time()
            print(f"opening {root} folder")
            true = detect_number(*get_true(f_name, df))
            results,results2,readtime = get_results(root, f_name)
            pred = detect_number2(*get_pred(results,results2))
            y_true.append(true); y_pred.append(pred)
            print(f"{f_name} is completed")
            time_p2 = time.time()
            proccesstime = time_p2 - time_p1
            time_for_sample += proccesstime
            time_for_read += readtime
            count += 1


print(f"need {(time_for_sample/count) * 1000} ms for proccessing one sample")
print(f"need {(time_for_read/count) * 1000} ms for read one sample")
###################
#####結果一覧#######
####################

# classes = ("negative","positive","need to review","Undecidable")
# y_true = np.array(y_true)
# y_pred = np.array(y_pred)
# cf_matrix = confusion_matrix(y_true, y_pred)
# df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
#                      columns = [i for i in classes])
# df_cm.to_csv("cm.csv")
# plt.figure(figsize = (20,9))
# sn.set(font_scale = 2.)
# sn.heatmap(df_cm, annot=True,fmt='g')
# plt.savefig('absolutely_value.png')
# plt.close()
# df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*len(classes), index = [i for i in classes],
#                      columns = [i for i in classes])
# sn.heatmap(df_cm, annot=True)
# plt.savefig("percentage.png")
