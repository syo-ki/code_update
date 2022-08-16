# -*- coding: utf-8 -*-
"""
Tspotのデータ前処理関数
パスを関数内に直書きしているので各々修正が必要
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2,os
import shutil
from PIL import Image



def make_samesize_image_for_truespot(img_size=(64,64)):
    """
    ファイル名にスポットの位置情報がある画像パスから画像を作り直す
    画像サイズは64×64で固定とする
    Parameters
    ----------
    dates:list
            対象日付の一覧
    root_path:str
            256*256にしたウェル画像フォルダへの共通パス、直後に日付が入る
    filepath:str
            spot画像へのパス

    Returns
    -------

    """
    dates=os.listdir(r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\well_png_256")
    root_path=r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\well_png_256"
    filespath=r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\spot\true_spot"

    filepaths=os.listdir(filespath)
    for i in range(len(filepaths)):
        print(i,filepaths[i])
        filename=filepaths[i]
        path_split=filename.split("_")
        plateNo=path_split[0]
        wellNo=path_split[1]
        pos=[int(int(p.replace(".png",""))*256/518) for p in  path_split[2:6]]
        date=""
        for d in dates:
            platepath=root_path+r"\%s"%(d)
            if plateNo in os.listdir(platepath):
                date=d
                break

        if date=="":
            continue

        img_path=root_path+r"\%s\%s\%s.png"%(date,plateNo,wellNo)

        img=cv2.imread(img_path)
        blank_val=255
        img_blank=np.zeros((64+256,64+256,3), np.uint8)

        img_blank.fill(blank_val)
        img_merge=np.copy(img_blank)

        img_merge[32:256+32,32:256+32,:]=img


        center=[int((pos[3]+pos[1])/2),int((pos[2]+pos[0])/2)]
        img_spot=img_merge[center[0]-32+32:center[0]+32+32,center[1]-32+32:center[1]+32+32]
        cv2.imwrite(r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\spot\true_spot_64_64\%s"%(filename),img_spot)
        """cv2.imshow('ori_image',img)
        show_image(img_spot)"""


def make_samesize_image_for_redspot(img_size=(64,64)):
    """
    ファイル名にスポットの位置情報がある画像パスから画像を作り直す
    画像サイズは64×64で固定とする
    """
    dates=os.listdir(r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\well_png_256")
    root_path=r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\well_png_256"
    filespath=r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\spot\red_spot"
    filepaths=os.listdir(filespath)
    for i in range(len(filepaths)):
        print(i,filepaths[i])
        filename=filepaths[i]
        path_split=filename.split("_")
        plateNo=path_split[1]
        wellNo=path_split[2]
        pos=[int(int(p.replace(".png",""))*256/518) for p in  path_split[3:]]
        pos=[pos[0]-0,pos[1]-0,pos[0]+48,pos[1]+48]
        date=""
        for d in dates:
            platepath=root_path+r"\%s"%(d)
            if plateNo in os.listdir(platepath):
                date=d
                break

        if date=="":
            continue

        img_path=root_path+r"\%s\%s\%s.png"%(date,plateNo,wellNo)

        img=cv2.imread(img_path)
        blank_val=255
        img_blank=np.zeros((64+256,64+256,3), np.uint8)

        img_blank.fill(blank_val)
        img_merge=np.copy(img_blank)

        img_merge[32:256+32,32:256+32,:]=img


        center=[int((pos[3]+pos[1])/2),int((pos[2]+pos[0])/2)]
        img_spot=img_merge[center[0]-32+32:center[0]+32+32,center[1]-32+32:center[1]+32+32]
        cv2.imwrite(r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\spot\red_spot_64_64\%s"%(filename),img_spot)
        """cv2.imshow('ori_image',img)
        show_image(img_spot)"""


def show_image(img,showname="show_image"):
    """
    確認用関数。受け取った画像を表示する
    Parameters
    ----------
    img:array
        画像配列
    showname:str
        表示時の窓名
    """
    cv2.imshow(showname,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def copyxml():
    """
    アノテーションにより作成したxmlファイルを一つのフォルダ下にコピーして入れる
    Parameters
    ----------
    path_product:sr
            作成物の置き場所
    rootPath:str
        対象の画像への共通パス

    """
    path_product=r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\labelImg-master\product\xml"
    rootPath=r"C:\Users\srl_kono\OneDrive - Miraca Holdings Inc\TSPOT_DATA\well_png"
    folder_dates=os.listdir(rootPath)
    for date in folder_dates:
        print(date)
        path_date=rootPath+r"\%s"%(date)
        folder_plates=os.listdir(path_date)
        for plate in folder_plates:
            path_plate=path_date+r"\%s"%(plate)
            if os.path.isfile(path_plate):
                continue
            try:
                folder_files=os.listdir(path_plate)
            except:
                continue

            for file in folder_files:
                base, ext = os.path.splitext(file)#ファイル名を本文と拡張子に分ける
                if ext==".xml":
                    print(base,ext)
                    tgtName=path_plate+"\\"+file
                    saveName=path_product+"\\"+"%s_%s_%s"%(date,plate,file)
                    if not os.path.exists(saveName):
                        shutil.copy(tgtName,saveName)



def auto_detect_spots():
    """
    自動機の判定情報をlabelImgで使用できる形式で作成する
    自動機の検出結果であるDetails.xmlを参照
    各ウェルを参照する。この際ポジコンであるD行、H行は参照しない。またネガコンであるA行、E行にスポットが10個以上ある場合も参照しない
    各ウェルにスポットタグが存在する場合にそのスポット情報を取り出し、ウェル画像上でその位置を確認する。その位置が青スポットの場合は位置情報をスポットとして保存する
    赤スポットの場合は非スポットとして位置情報を保存する
    Parameters
    ----------
    blackWells:array
            プレート画像の白部分を黒くした画像
    start_pos:tuple
            プレートのA1ウェルの左上座標
    circle_size:tuple
            プレート画像上の1つ分のwellのサイズ
    dates: list
            日付一覧のリスト。対象の日付のみを入れる
    rootPath:str
            抽出画像データへの共通パス
    csvfilerootPath:str
            抽出カウントデータへの共通パス
    Returns
    -------

    """
    import xml.etree.ElementTree as ET
    #plate全体の黒塗り画像
    blackWells=cv2.imread("C:\\Users\\srl_kono\\Anaconda3\\test_folder\\tspot\\allwell_black.png")
    #画像外枠を白塗りにする
    blackWells[::,:96]=(255,255,255)
    blackWells[::,2505:]=(255,255,255)
    blackWells[:200,::]=(255,255,255)
    blackWells[1843:,::]=(255,255,255)
    start_pos=(96,200)
    circle_size=(201,205)


    pos_x=[x*circle_size[0]+start_pos[0] for x in range(12)]#ウェルx座標
    pos_y=[y*circle_size[1]+start_pos[1] for y in range(8)] #ウェルy座標
    ABC=["A","B","C","D","E","F","G","H"]
    rootPath="G:\\T-spot\\analysis_data\\"
    csvfilerootPath="G:\\T-spot\\export_data\\"
    #dates=os.listdir(rootPath)

    dates=['2018-12-06', '2018-12-07', '2018-12-08', '2018-12-09', '2018-12-10', '2018-12-12',
    	'2018-12-13', '2018-12-14', '2018-12-15', '2018-12-16', '2018-12-17', '2019-04-03',
    	'2019-04-04', '2019-04-05', '2019-04-06', '2019-04-07', '2019-04-08']

    count_test=0

    for d in dates:
        count_test+=1
        print("Date:",d)
        d_path=rootPath+d
        plates=os.listdir(d_path)
        for plate in plates:
            print("plate:",plate)
            plate_path=d_path+"\\"+str(plate)
            if os.path.isfile(plate_path):#対象プレートが存在しない場合は次の繰り返しへ
                print("this plate do not exist")
                continue
            l_im=None
            plate_image_path=r"G:\T-spot\analysis_data\%s\%s\qc\composite.png"%(d,plate)

            if not os.path.exists(plate_image_path):#検査員判定のある全体画像がない場合は次の繰り返しへ
                print("composite image do not exist")
                continue
            l_im=cv2.imread(plate_image_path)
            detail_xml=plate_path+"\\counted\\Details.xml"
            if not os.path.exists(detail_xml):#自動機判定がない場合は次の繰り返しへ
                print("detail.xml do not exist")
                continue

            splitdate=d.split("-")
            year,month,day=splitdate[0],splitdate[1],splitdate[2]
            candidates=["%s%s%s"%(year,month,str(int(day)+i).zfill(2) ) for i in range(5)]
            csvfile=None
            csvflag=0
            for candidate in candidates:
                csvfilepath=csvfilerootPath+"%s\\%s.csv"%(candidate,plate)
                if os.path.exists(csvfilepath):
                    csvfile=np.loadtxt(csvfilepath,delimiter=",",encoding="utf-8",dtype=str)
                    csvflag=1
                    break

            if csvflag==0:#csvfileがない場合を除外
                print(" Csv file does not exist")
                continue


            try:
                tree=ET.parse(detail_xml)
            except:
                print("Parse error")
                continue
            treeroot=tree.getroot()
            CountedWells=treeroot.find("CountedWells")
            print("Date:",d,", Plate:",plate)

            for CountedWell in CountedWells:
                Column=int(CountedWell.find("Column").text)
                Row=int(CountedWell.find("Row").text)

                print("Column(横)：",Column,"Row(縦)：",ABC[Row-1])#A1,B1,C1。。。,の順,col=1~12,,row=1~8(A~H)
                if Row in [4,8]:#ポジティブコントロールを除外
                    print("This is positive control")
                    continue

                WellResult=CountedWell.find("WellResult")
                ActualSpotCount=WellResult.find("ActualSpotCount").text#自動機のカウント数
                print("ActualSpotCount:",ActualSpotCount)
                if int(ActualSpotCount)==0:#スポット数０を除外
                    print("machine judges this well do not have any spot")
                    continue

                index=np.where(csvfile[:,2]=="%s%s"%(ABC[Row-1],Column))[0][0]
                tgt_csv=csvfile[index,:]
                try:
                    true_spot_count=int(tgt_csv[4])
                except:
                    print("true spot count includes str ,for example'>20'")
                    continue
                if true_spot_count==0:
                    print("expert judges this well do not have any spot")
                    continue
                if int(Row) in [1,5]:#ネガティブコントロールの場合
                    if true_spot_count>=10:
                        print("negative controll includes over spots")
                        continue
                else:#NC以外
                    negative_Row=1 if Row<5 else 5
                    negative_index=(Column-1)*8+int(negative_Row)
                    negative_spot_count=int(csvfile[negative_index,4])
                    if negative_spot_count>=10:#参照すべきNCのスポットカウントが10以上なら除外
                        print("negative controll includes over spots")
                        continue

                Spot_List=WellResult.find("Spot_List")
                centers=[]
                for Spot in Spot_List:
                    Outline=Spot.find("Outline")
                    Outline_Length=int(Outline.attrib["Length"])
                    Cx,Cy=0,0
                    for P in Outline:
                        Cx+=int(P.attrib["X"])
                        Cy+=int(P.attrib["Y"])
                    Cx,Cy=int(Cx/Outline_Length),int(Cy/Outline_Length)
                    Cx,Cy=float(Cx/518.0),float(Cy/518.0)
                    centers.append([Cx,Cy])

                l_im=np.where(blackWells==(255,255,255),blackWells,(l_im))

                x_lt=pos_x[Column-1]
                y_lt=pos_y[Row-1]
                im_well=l_im[y_lt:y_lt+circle_size[1] ,x_lt:x_lt+circle_size[0]]
                im_well_copy=np.copy(im_well)


                a=176 if Column<8 else 170
                b=178 if Row <6 else 185
                spot_positions=[]
                for center in centers:
                    upper_lim=int(center[1]*b-10) if int(center[1]*b-10)>=0 else 0
                    lower_lim=int(center[1]*b+10) if int(center[1]*b+10)<=im_well.shape[1] else im_well.shape[1]
                    left_lim=int(center[0]*a-10) if int(center[0]*a-10)>=0 else 0
                    right_lim=int(center[0]*a+10) if int(center[0]*a+10)<=im_well.shape[0] else im_well.shape[0]
                    im_spot=im_well[upper_lim:lower_lim,left_lim:right_lim]
                    try:
                        img_mask_blue = cv2.inRange(im_spot, np.array([200, 0, 0]),np.array([255, 100, 100])) # BGRからマスクを作成
                        index_blue=np.where(img_mask_blue!=0)
                        img_mask_green = cv2.inRange(im_spot, np.array([0, 100, 0]),np.array([100, 255, 100])) # BGRからマスクを作成
                        index_green=np.where(img_mask_green!=0)
                        img_mask_red = cv2.inRange(im_spot, np.array([0, 0, 200]),np.array([100, 100, 255])) # BGRからマスクを作成
                        index_red=np.where(img_mask_red!=0)
                        print("--------------SPOT----------------")
                        print("positon:",center)
                        print("B:",index_blue)
                        print("G:",index_green)
                        print("R:",index_red)
                        print("----------------------------------")
                        """

                        場合わけ　目標範囲内の色
                        　赤点があり　：非スポットなのでカウントしない
                        　青点のみ　：スポットとしてカウント
                        　緑：スポットとして一応カウント
                        ひとつでもスポットとしてカウントする場合にlabelimgと同様のxmlファイルを作成する

                        """

                        if len(index_red[0])>=3:#赤点がある
                            print("This is red spot")
                            continue
                        if len(index_blue[0]>=3) or len(index_green[0]>=3):#青あるいは緑点あり
                            cs_x=450
                            cs_y=450
                            spot_positions.append([int(center[0]*cs_x-15),int(center[1]*cs_y-15),int(center[0]*cs_x+15),int(center[1]*cs_y+15)])
                    except:
                        continue

                if len(spot_positions)>=1:
                    FileName=str(ABC[Row-1])+str(Column)+".png"
                    Path=r"C:\Users\srl_kono\OneDrive - Miraca Holdings Inc\TSPOT_DATA\well_png"+"\\"+d+"\\"+plate+"\\"+FileName
                    try:
                        make_xml(Plate=str(plate),FileName=FileName,Path=Path,spots=spot_positions)
                        print("Path:",Path)
                    except:
                        print("error occurs")
                        continue
                else:
                    print("No spot exists")






def make_xml(Plate=None,FileName=None,Path=None,spots=None,overwrite=False):
    """
    lblImg様のxmlファイルを作成する,すでにある場合は保存しない

    Parameters
    ----------
    Plate:str
            プレート番号。フォルダ名の入力に用いる
    Filename:str
            ファイル名。フォルダ名の入力に用いる
    Path:str
            作成するファイル名
    spots:list
            入力するスポット群
    overwrite:bool
            ファイル上書きの可否

    """
    from xml.etree.ElementTree import Element, SubElement
    import xml.etree.ElementTree as ET
    annotation=Element("annotation")
    folder=SubElement(annotation,"folder")
    folder.text=Plate
    filename=SubElement(annotation,"filename")
    filename.text=FileName
    path=SubElement(annotation,"path")
    path.text=Path
    source=SubElement(annotation,"source")
    database=SubElement(source,"database")
    database.text="Unknown"
    size=SubElement(annotation,"size")
    width=SubElement(size,"width")
    width.text=str(0)
    height=SubElement(size,"height")
    height.text=str(0)
    depth=SubElement(size,"depth")
    depth.text=str(3)
    segmented=SubElement(annotation,"segmented")
    segmented.text=str(0)

    for spot in spots:
        obj=SubElement(annotation,"object")
        name=SubElement(obj,"name")
        name.text="spot"
        pose=SubElement(obj,"pose")
        pose.text="Unspecified"
        truncated=SubElement(obj,"truncated")
        truncated.text=str(0)
        difficult=SubElement(obj,"difficult")
        difficult.text=str(0)
        bndbox=SubElement(obj,"bndbox")
        xmin=SubElement(bndbox,"xmin")
        xmin.text=str(spot[0])
        ymin=SubElement(bndbox,"ymin")
        ymin.text=str(spot[1])
        xmax=SubElement(bndbox,"xmax")
        xmax.text=str(spot[2])
        ymax=SubElement(bndbox,"ymax")
        ymax.text=str(spot[3])

    tree = ET.ElementTree(annotation)
    indent(annotation)
    if not os.path.exists(Path.replace(".png",".xml")) or overwrite==True:
        tree.write(Path.replace(".png",".xml"), encoding='UTF-8')



def indent(elem, level=0):
    """
    xmlにインデントを追加する
    Parameters
    ----------
    elem:Element()
        xmlデータ
    level:int
        インデントの深さ
    """
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def get_spot_img_from_xml():
	"""
	lblImgにより作成したxmlよりスポット情報を抽出し、スポットの画像を得る
    Parameters
    ----------
    xmlpath:str
            対象のxmlファイル置き場
    savefolderpath:str
            作成物の保存フォルダパス
        　
	"""

	import xml.etree.ElementTree as ET
	xmlpath=r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\labelImg-master\product\xml"
	savefolderpath="C:\\Users\\srl_kono\\Anaconda3\\test_folder\\tspot\\labelImg-master\\product\\spot_from_xml\\"
	files=os.listdir(xmlpath)
	startflg=0
	for f in files:
		tree = ET.parse(xmlpath+ "\\"+f)
		root=tree.getroot()
		filename=""
		filepath=""
		annotation=0
		xmin,xmax,ymin,ymax= -99,-99,-99,-99
		spots=[]
		for i in tree.iter():
			print(i.tag, i.attrib, i.text)
			annotation+=1 if i.tag=="annotation" else annotation
			filename=i.text if i.tag=="filename" else filename
			filepath=i.text if i.tag=="path" else filepath
			filepath=filepath.replace("C:\\Users\\srl\\Anaconda3\\envs\\py36\\sourcecode\\tspot\\producted\\",
									"C:\\Users\\srl_kono\\OneDrive - Miraca Holdings Inc\\TSPOT_DATA\\")

			xmin=int(i.text) if i.tag=="xmin" else xmin
			ymin=int(i.text) if i.tag=="ymin" else ymin
			xmax=int(i.text) if i.tag=="xmax" else xmax
			ymax=int(i.text) if i.tag=="ymax" else ymax
			if i.tag=="ymax" :
				spots.append([xmin,ymin,xmax,ymax])

		if not os.path.exists(filepath):
			continue
		else:
			im_well=cv2.imread(filepath)
			for spot in spots:
				im_spot=im_well[spot[1]:spot[3],spot[0]:spot[2]]
				path_split=filepath.split("\\")
				savename=savefolderpath+ str(path_split[-2])+"-"+str(path_split[-1].replace(".png",""))+"_"+str(spot[0]) +"_"+str(spot[1])+".png"
				cv2.imwrite(savename,im_spot)
		startflg=1


def split_xmls_fot_3types(tr_val_te_rate=[0.7,0.2,0.1]):
	"""
	アノテーションツールにより作成したxmlファイル群をtrain,validation,testの３つにランダムに分ける
	分けたファイル名（拡張子を除く）をテキストファイルで保存する
	train、validation：学習用,test:評価用
    Parameters
    ----------
    tr_val_te_rate:list
            train,validation,testの分ける比率
    xmlpath:str
            処理対象のxmlファイル置き場
	"""
	xmlpath=r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\labelImg-master\product\xml"
	xmls=np.array(os.listdir(xmlpath))
	num_all=len(xmls)
	num_train=int(num_all*tr_val_te_rate[0])
	num_test=int(num_all*tr_val_te_rate[1])
	num_val=num_all-num_train-num_test
	id_all   = np.random.choice(num_all, num_all, replace=False)
	id_te  = id_all[0:num_test]
	id_tr = id_all[num_test:num_test+num_train]
	id_val=id_all[num_test+num_train:num_test+num_all]
	te_data  = xmls[id_te]
	tr_data = xmls[id_tr]
	val_data= xmls[id_val]

	te_data=np.array([v.replace(".xml","") for v in te_data])
	tr_data=np.array([v.replace(".xml","") for v in tr_data])
	val_data=np.array([v.replace(".xml","") for v in val_data])

	f = open(r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\keras-yolo3-master\VOCDevkit\VOC2007\ImageSets\Main\train.txt", 'w')
	for x in tr_data:
	    f.write(str(x) + "\n")
	f.close()
	f = open(r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\keras-yolo3-master\VOCDevkit\VOC2007\ImageSets\Main\test.txt", 'w')
	for x in te_data:
	    f.write(str(x) + "\n")
	f.close()
	f = open(r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\keras-yolo3-master\VOCDevkit\VOC2007\ImageSets\Main\validation.txt", 'w')
	for x in val_data:
	    f.write(str(x) + "\n")
	f.close()

def copy_well_image():
	"""
	textファイル内に記載された画像を指定のサイズで複製する
    Parameters
    ----------
    savePath:str
            画像の保存先
    textfile:str
            処理対象の名前を記述したテキストファイルのパス
    imgPath:str
            コピー元の画像フォルダ
    img_shape:tuple
            作成物のサイズ
	"""
	savePath=r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\keras-yolo3-master\VOCDevkit\VOC2007\Images\\"
	#txtfile=r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\keras-yolo3-master\VOCDevkit\VOC2007\ImageSets\Main\train.txt"
	#txtfile=r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\keras-yolo3-master\VOCDevkit\VOC2007\ImageSets\Main\val.txt"
	txtfile=r"C:\Users\srl_kono\Anaconda3\test_folder\tspot\keras-yolo3-master\VOCDevkit\VOC2007\ImageSets\Main\test.txt"

	imgPath=r"C:\Users\srl_kono\OneDrive - Miraca Holdings Inc\TSPOT_DATA\well_png\\"
	tgts=[]
	img_shape=(416,416)
	f = open(txtfile,"r")
	for x in f:
		tgts.append(x.rstrip("\n"))
	f.close()
	for tgt in tgts:
		tgt_split=tgt.split("_")
		filename=imgPath+tgt_split[0]+"\\"+tgt_split[1]+"\\"+tgt_split[2]+".png"
		if os.path.exists(filename):
			savename=savePath+tgt+".png"
			im=cv2.imread(filename)
			im=cv2.resize(im,img_shape)
			cv2.imwrite(savename,im)



def get_rotate_image(): #未完成
	"""
	すでにラベルとして作成したテキストファイルから画像を読み込み、指定の角度回転させた画像、スポット情報を得る
	Parameters
    ----------
    degree:int
            回転角度
    im_shape:tuple
            画像サイズ
    txtpath:str
            処理対象の画像名を記載したテキストファイルのパス

	"""
	degree=270#回転させたい角度
	im_shape=(416,416,3)
	txtpath=r"C:\\Users\\srl_kono\\Anaconda3\\test_folder\\tspot\\keras-yolo3-master\\2007_train.txt"
	with open(txtpath) as f:
		lines = f.readlines()

	for l in lines:
		l_split=l.split(" ")
		filename=l_split[0]
		spots=[s.replace("\n","") for s in  l_split[1:]]
		print(filename)
		img_afn=rotation_cv2(filename,-degree)
		img_afn = cv2.cvtColor(img_afn, cv2.COLOR_BGR2RGB)
		spots_afn=[]
		for spot in spots:
			spot_pos=spot.split(",")
			center=[int( (int(spot_pos[2])+int(spot_pos[0]) )/2)-int(im_shape[0]/2),int( (int(spot_pos[3])+int(spot_pos[1]) )/2)-int(im_shape[1]/2)]
			w=int(int(spot_pos[2])-int(spot_pos[0]))
			h=int(int(spot_pos[3])-int(spot_pos[1]))
			new_center=[int(center[0]*np.cos(np.radians(degree))-center[1]*np.sin(np.radians(degree))+int(im_shape[0]/2)),
						int(center[0]*np.sin(np.radians(degree))+center[1]*np.cos(np.radians(degree))+int(im_shape[1]/2))]

			print("new",new_center)
			spots_afn.append([int(new_center[0]-w/2),int(new_center[1]-h/2),int(new_center[0]+w/2), int(new_center[1]+h/2),0])
			#cv2.rectangle(img_afn, (int((new_center[0]-w/2)), int((new_center[1]-h/2))), (int((new_center[0]+w/2)), int((new_center[1]+h/2))), (255, 0, 0))
		print(spots_afn)

		#show_image(img_afn)
		print(stop)






	pass



def rotation_cv2(path,degree):
    img = np.asarray(Image.open(path), dtype=np.uint8)

    center = (img.shape[1]//2, img.shape[0]//2)
    affine = cv2.getRotationMatrix2D(center, degree,  1.0)
    img_afn = cv2.warpAffine(img, affine, (img.shape[1], img.shape[0]),flags=cv2.INTER_LINEAR,borderValue=(255, 255, 255))

    #plt.imshow(img_afn)
    #plt.show()
    return img_afn

def create_png_composite():
    dates=["2022-05-23","2022-05-25","2022-05-26","2022-05-27","2022-05-28","2022-05-29","2022-05-30","2022-06-01",
           "2022-06-02","2022-06-03","2022-06-04","2022-06-05","2022-06-06","2022-06-08","2022-06-09","2022-06-10",
           "2022-06-11","2022-06-12","2022-06-13","2022-06-15","2022-06-16","2022-06-17","2022-06-18","2022-06-19",
           "2022-06-20"]
    import shutil
    for date in dates:
        path = f"./analysis_data/{date}"
        plate_number = os.listdir(path)
        for plate_n in plate_number:
            p = os.path.join(path, plate_n)
            file_name = os.listdir(p)
            for f_n in file_name:
                if f_n[-5:-1] == ".CTL":
                    read_p = os.path.join(p, f_n)
                    new_f_n = read_p.replace(".CTL",".png")
                    save_p = f"./well_png/{date}/{new_f_n}"
                    shutil.copyfile(read_p, save_p)
            read_p = os.path.join(p,"qc/composite.png")
            if os.path.exists(read_p):
                save_p = f"./composite_img/{date}/{plate_n} + .png"
                shutil.copyfile(read_p, save_p)
            else:
                print(f"can't find composite image of {plate_n}")






if __name__ == '__main__':
    import sys
    import os
    sys.path.append('.')
    sys.path.append(os.pardir)

    #make_samesize_image_for_redspot()
    #copyxml()
    #auto_detect_spots()
    #make_xml()
    #get_spot_img_from_xml()
    #split_xmls_fot_3types()
    #copy_well_image()
    # get_rotate_image()
    create_png_composite()
