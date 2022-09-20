"""This module is a libray of the coordinates used for reconstruction"""

#Mouse setting 2020-09-22
x_0=int(-10)
y_0=int(4)

y1_front=y_0+460
y2_front=y_0+646
x1_front=x_0+450
x2_front=x_0+602

y1_top=y_0+699
y2_top=y_0+860
x1_top=x_0+457
x2_top=x_0+609

y1_120=y_0+703
y2_120=y_0+864
x1_120=x_0+214
x2_120=x_0+366

y1_240=y_0+703
y2_240=y_0+864
x1_240=x_0+690
x2_240=x_0+842


#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,0.45,[y1_front,y2_front,x1_front,x2_front],None,None,None,False,10/9).image     #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,0.45,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,0.45,[y1_120,y2_120,x1_120,x2_120],None,None,None,False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,0.45,[y1_240,y2_240,x1_240,x2_240],None,None,None,False,10/9).image





#high dose setting 2020-09-22 top ,120 and 240 have equal szart and end y point  deltaz=155



x_0=int(-10)
y_0=int(0)

y1_front=y_0+460
y2_front=y_0+646
x1_front=x_0+450
x2_front=x_0+602

y1_top=y_0+702
y2_top=y_0+857
x1_top=x_0+448
x2_top=x_0+600

y1_120=y_0+702
y2_120=y_0+857
x1_120=x_0+206
x2_120=x_0+358

y1_240=y_0+702
y2_240=y_0+857
x1_240=x_0+690
x2_240=x_0+842

#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,0.2,[y1_front,y2_front,x1_front,x2_front],None,None,None,False,10/9).image     #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,0.2,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,0.2,[y1_120,y2_120,x1_120,x2_120],None,None,None,False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,0.2,[y1_240,y2_240,x1_240,x2_240],None,None,None,False,10/9).image





#high dose 2020-09-22 deltaz=161


x_0=int(-10)
y_0=int(0)

y1_front=y_0+460
y2_front=y_0+646
x1_front=x_0+450
x2_front=x_0+602

y1_top=y_0+696
y2_top=y_0+857
x1_top=x_0+448
x2_top=x_0+600

y1_120=y_0+702
y2_120=y_0+863
x1_120=x_0+206
x2_120=x_0+358

y1_240=y_0+702
y2_240=y_0+863
x1_240=x_0+690
x2_240=x_0+842

#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,0.2,[y1_front,y2_front,x1_front,x2_front],None,None,None,False,10/9).image     #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,0.2,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,0.2,[y1_120,y2_120,x1_120,x2_120],None,None,None,False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,0.2,[y1_240,y2_240,x1_240,x2_240],None,None,None,False,10/9).image

#im=plt.imread(path_pic_front)









x_0=int(-15)
y_0=int(4)


y1_front=y_0+466
y2_front=y_0+652
x1_front=x_0+451
x2_front=x_0+603

y1_top=y_0+700
y2_top=y_0+859
x1_top=x_0+451
x2_top=x_0+603

y1_120=y_0+703
y2_120=y_0+862
x1_120=x_0+208
x2_120=x_0+360

y1_240=y_0+701
y2_240=y_0+860
x1_240=x_0+694
x2_240=x_0+846

#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,0.4,[y1_front,y2_front,x1_front,x2_front],None,None,None,False,10/9).image     #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,0.4,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,0.4,[y1_120,y2_120,x1_120,x2_120],None,None,None,False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,0.4,[y1_240,y2_240,x1_240,x2_240],None,None,None,False,10/9).image


#low dose 2020-09-25 deltaz=159
shape_front=(170,152) #you can not reconstruct somethin bigger than 161
shape_side=(159,152)


rot_angle=0.4


x_0=int(-60)
y_0=int(5)

y1_front=y_0+466
y2_front=y_0+636
x1_front=x_0+453
x2_front=x_0+605

y1_top=y_0+697
y2_top=y_0+856
x1_top=x_0+453
x2_top=x_0+605

y1_120=y_0+701
y2_120=y_0+860
x1_120=x_0+208
x2_120=x_0+360

y1_240=y_0+702
y2_240=y_0+861
x1_240=x_0+693
x2_240=x_0+845



###shift
shift_image_front =[0,0]
shift_image_top = [0,0]
shift_image_120 = [0,0]
shift_image_240 = [5,0]



mask_border_front=[0,0,0,0]
mask_border_top=[0,0,0,0]
mask_border_120=[5,0,0,0]
mask_border_240=[10,5,0,0]





#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,rot_angle,[y1_front,y2_front,x1_front,x2_front],None,None,shift_image_front,mask_border_front,False,10/9).image  #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,rot_angle,[y1_top,y2_top,x1_top,x2_top],None,None,shift_image_top,mask_border_top,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,rot_angle,[y1_120,y2_120,x1_120,x2_120],None,None,shift_image_120,mask_border_120,False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,rot_angle,[y1_240,y2_240,x1_240,x2_240],None,None,shift_image_240,mask_border_240,False,10/9).image












#high dose 2020-09-24 deltaz=159

x_0=int(-23)    #they change for each images of this day
y_0=int(4)


y1_front=y_0+466
y2_front=y_0+652
x1_front=x_0+451
x2_front=x_0+603

y1_top=y_0+699
y2_top=y_0+858
x1_top=x_0+451
x2_top=x_0+603

y1_120=y_0+704
y2_120=y_0+863
x1_120=x_0+208
x2_120=x_0+360

y1_240=y_0+702
y2_240=y_0+861
x1_240=x_0+694
x2_240=x_0+846

#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,0.5,[y1_front,y2_front,x1_front,x2_front],None,None,None,False,10/9).image     #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,0.5,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,0.5,[y1_120,y2_120,x1_120,x2_120],None,None,None,False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,0.5,[y1_240,y2_240,x1_240,x2_240],None,None,None,False,10/9).image


#laser first measure lots of holes
#spatial resolution laser  deltax=175



x_0=int(28)
y_0=int(-5)

y1_front=y_0+460
y2_front=y_0+646
x1_front=x_0+432
x2_front=x_0+600

y1_top=y_0+696
y2_top=y_0+857
x1_top=x_0+432
x2_top=x_0+600

y1_120=y_0+700
y2_120=y_0+861
x1_120=x_0+200
x2_120=x_0+368

y1_240=y_0+702
y2_240=y_0+863
x1_240=x_0+662
x2_240=x_0+830

#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,89.2,[y1_front,y2_front,x1_front,x2_front],None,None,None,False,10/9).image     #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,89.2,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,89.2,[y1_120,y2_120,x1_120,x2_120],None,None,None,False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,89.2,[y1_240,y2_240,x1_240,x2_240],None,None,None,False,10/9).image










#spatial resolution laser  deltaz=159 obliquo 3 holes
x_0=int(-45)
y_0=int(-6)

y1_front=y_0+460
y2_front=y_0+646
x1_front=x_0+438
x2_front=x_0+590

y1_top=y_0+700
y2_top=y_0+871
x1_top=x_0+433
x2_top=x_0+592

y1_120=y_0+699
y2_120=y_0+870
x1_120=x_0+202
x2_120=x_0+361

y1_240=y_0+702
y2_240=y_0+873
x1_240=x_0+664
x2_240=x_0+823

#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,89.9,[y1_front,y2_front,x1_front,x2_front],None,None,None,False,10/9).image     #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,89.9,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,89.9,[y1_120,y2_120,x1_120,x2_120],None,None,None,False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,89.9,[y1_240,y2_240,x1_240,x2_240],None,None,None,False,10/9).image





#spatial resolution laser  deltax=162 orizontal 10 holes

x_0=int(-45)
y_0=int(-6)

y1_front=y_0+460
y2_front=y_0+646
x1_front=x_0+440
x2_front=x_0+590

y1_top=y_0+700
y2_top=y_0+871
x1_top=x_0+436
x2_top=x_0+586

y1_120=y_0+700
y2_120=y_0+871
x1_120=x_0+190
x2_120=x_0+340

y1_240=y_0+702
y2_240=y_0+873
x1_240=x_0+676
x2_240=x_0+826

#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,89.7,[y1_front,y2_front,x1_front,x2_front],None,None,None,False,10/9).image     #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,89.7,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,89.7,[y1_120,y2_120,x1_120,x2_120],None,None,None,False,7/5).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,89.7,[y1_240,y2_240,x1_240,x2_240],None,None,None,False,7/5).image














#spatial resolution laser  deltax=162 vertical 7 holes



x_0=int(-44)
y_0=int(-6)

y1_front=y_0+460
y2_front=y_0+646
x1_front=x_0+440
x2_front=x_0+590

y1_top=y_0+700
y2_top=y_0+871
x1_top=x_0+436
x2_top=x_0+586

y1_120=y_0+700
y2_120=y_0+871
x1_120=x_0+195
x2_120=x_0+345

y1_240=y_0+703
y2_240=y_0+874
x1_240=x_0+676
x2_240=x_0+826

#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,89.65,[y1_front,y2_front,x1_front,x2_front],None,None,None,False,15/10).image     #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,89.65,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,89.65,[y1_120,y2_120,x1_120,x2_120],None,None,None,False,7/5).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,89.65,[y1_240,y2_240,x1_240,x2_240],None,None,None,False,7/5).image











#high dose dose 2020-11-05 deltaz=159



x_0=int(20)
y_0=int(-58)

y1_front=y_0+460
y2_front=y_0+646
x1_front=x_0+453
x2_front=x_0+605

y1_top=y_0+696
y2_top=y_0+855
x1_top=x_0+453
x2_top=x_0+605

y1_120=y_0+692
y2_120=y_0+851
x1_120=x_0+208
x2_120=x_0+360

y1_240=y_0+690
y2_240=y_0+849
x1_240=x_0+693
x2_240=x_0+845

#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,0,[y1_front,y2_front,x1_front,x2_front],None,None,None,False,10/9).image     #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,0,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,0,[y1_120,y2_120,x1_120,x2_120],None,None,None,False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,0,[y1_240,y2_240,x1_240,x2_240],None,None,None,False,10/9).image




#low dose  2020-11-05 deltaz=159



x_0=int(-20)
y_0=int(-58)

y1_front=y_0+455
y2_front=y_0+641
x1_front=x_0+453
x2_front=x_0+605

y1_top=y_0+697
y2_top=y_0+856
x1_top=x_0+451
x2_top=x_0+603

y1_120=y_0+694
y2_120=y_0+853
x1_120=x_0+210
x2_120=x_0+362

y1_240=y_0+694
y2_240=y_0+853
x1_240=x_0+693
x2_240=x_0+845

#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,0,[y1_front,y2_front,x1_front,x2_front],None,None,None,False,10/9).image     #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,0,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,0,[y1_120,y2_120,x1_120,x2_120],None,None,None,False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,0,[y1_240,y2_240,x1_240,x2_240],None,None,None,False,10/9).image




#high dose  2020-12-03 deltaz=159



shape_front=(170,152) #you can not reconstruct somethin bigger than 161
shape_side=(159,152)


rot_angle=0
#rot_angle=.4
x_0=int(15)
y_0=int(-58)

y1_front=y_0+464
y2_front=y_0+634
x1_front=x_0+453
x2_front=x_0+605

y1_top=y_0+697
y2_top=y_0+856
x1_top=x_0+451
x2_top=x_0+603

y1_120=y_0+694
y2_120=y_0+853
x1_120=x_0+215
x2_120=x_0+367

y1_240=y_0+694
y2_240=y_0+853
x1_240=x_0+693
x2_240=x_0+845
x2_240=x_0+845



###shift
shift_image_front =[0,0]
shift_image_top = [0,0]
shift_image_120 = [0,0]
shift_image_240 = [0,0]



mask_border_front=[0,0,0,0]
mask_border_top=[0,0,0,0]
mask_border_120=[0,0,0,0]
mask_border_240=[10,0,0,0]


#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,0,[y1_front,y2_front,x1_front,x2_front],None,None,None,False,10/9).image     #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,0,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,0,[y1_120,y2_120,x1_120,x2_120],None,None,None,False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,0,[y1_240,y2_240,x1_240,x2_240],None,None,None,False,10/9).image





#lowdose dose  2020-12-03 deltaz=159
directory='pictures/2020-12-03/lowdose/'
picture_name='5.png'

#Define path for measured pictures
path_pic_front=directory+picture_name
path_pic_top=directory+picture_name
path_pic_120=directory+picture_name
path_pic_240=directory+picture_name

path_pic2_front=directory+picture2_name
path_pic2_top=directory+picture2_name
path_pic2_120=directory+picture2_name
path_pic2_240=directory+picture2_name




deltaz=159 #Draco measurements
s=0.0634
#s= 0.0634   #spatialresolution proton measurements
#deltaz=159# proton measurements
#X-Ray 2021-08-13 deltaz=208
shape_front=(170,152) #you can not reconstruct somethin bigger than 161
shape_side=(159,152)


rot_angle=0
#rot_angle=.4
x_0=int(15)
y_0=int(-58)

y1_front=y_0+464
y2_front=y_0+634
x1_front=x_0+453
x2_front=x_0+605

y1_top=y_0+697
y2_top=y_0+856
x1_top=x_0+451
x2_top=x_0+603

y1_120=y_0+694
y2_120=y_0+853
x1_120=x_0+215
x2_120=x_0+367

y1_240=y_0+694
y2_240=y_0+853
x1_240=x_0+693
x2_240=x_0+845
x2_240=x_0+845



###shift
shift_image_front =[0,0]
shift_image_top = [0,0]
shift_image_120 = [0,0]
shift_image_240 = [0,0]



mask_border_front=[0,0,0,0]
mask_border_top=[0,0,0,0]
mask_border_120=[0,0,0,0]
mask_border_240=[10,0,0,0]





#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,rot_angle,[y1_front,y2_front,x1_front,x2_front],None,None,shift_image_front,mask_border_front,False,10/9).image  #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,rot_angle,[y1_top,y2_top,x1_top,x2_top],None,None,shift_image_top,mask_border_top,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,rot_angle,[y1_120,y2_120,x1_120,x2_120],None,None,shift_image_120,mask_border_120,False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,rot_angle,[y1_240,y2_240,x1_240,x2_240],None,None,shift_image_240,mask_border_240,False,10/9).image


cam_pic2_front=Image_loader(path_pic2_front,color_channel,None,None,rot_angle,[y1_front,y2_front,x1_front,x2_front],None,None,shift_image_front,mask_border_front,False,10/9).image
cam_pic2_top=Image_loader(path_pic2_top,color_channel,None,None,rot_angle,[y1_top,y2_top,x1_top,x2_top],None,None,shift_image_top ,mask_border_top,False,1).image
cam_pic2_120=Image_loader(path_pic2_120,color_channel,None,1,rot_angle,[y1_120,y2_120,x1_120,x2_120],None,None,shift_image_120,mask_border_120,False,10/9).image
cam_pic2_240=Image_loader(path_pic2_240,color_channel,None,1,rot_angle,[y1_240,y2_240,x1_240,x2_240],None,None,shift_image_240,mask_border_240,False,10/9).image






#low dose  2021-11-03 deltaz=159 munich



x_0=int(-60)
y_0=int(40)

y1_front=y_0+435
y2_front=y_0+621
x1_front=x_0+453
x2_front=x_0+605

y1_top=y_0+725
y2_top=y_0+884
x1_top=x_0+451
x2_top=x_0+603

y1_120=y_0+725
y2_120=y_0+884
x1_120=x_0+180
x2_120=x_0+332

y1_240=y_0+727
y2_240=y_0+886
x1_240=x_0+743
x2_240=x_0+895

#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,270,[y1_front,y2_front,x1_front,x2_front],None,None,None,False,10/9).image     #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,270,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,270,[y1_120,y2_120,x1_120,x2_120],None,None,None,False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,270,[y1_240,y2_240,x1_240,x2_240],None,None,None,False,10/9).image









#######################################################################################################################################
#XRay measurement  2021-08-13 deltaz=202 shot 6




directory='pictures/2021-08-13/'
#directory='pictures/spatialresolution/'
picture_name='shot_6.png'
picture2_name='shot_9.png' #background
#picture_name='obliquel100ms100mgplusother_2020-10-08.png'

deltaz=202 #XRay measurements
s=0.0508
#s= 0.0634   #spatialresolution proton measurements
#deltaz=159# proton measurements
#X-Ray 2021-08-13 deltaz=208


rot_angle=180.4
x_0=int(50)
y_0=int(25)

y1_front=y_0+440
y2_front=y_0+650
x1_front=x_0+440
x2_front=x_0+620

y1_top=y_0+727
y2_top=y_0+929
x1_top=x_0+440
x2_top=x_0+620

y1_120=y_0+718
y2_120=y_0+920
x1_120=x_0+150
x2_120=x_0+330

y1_240=y_0+721
y2_240=y_0+923
x1_240=x_0+720
x2_240=x_0+900

#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,rot_angle,[y1_front,y2_front,x1_front,x2_front],None,None,[0,-5],False,10/9).image  #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,rot_angle,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,rot_angle,[y1_120,y2_120,x1_120,x2_120],None,None,[15,0],False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,rot_angle,[y1_240,y2_240,x1_240,x2_240],None,None,[-15,0],False,10/9).image


cam_pic2_front=Image_loader(path_pic2_front,color_channel,None,None,rot_angle,[y1_front,y2_front,x1_front,x2_front],None,None,[0,-5],False,10/9).image
cam_pic2_top=Image_loader(path_pic2_top,color_channel,None,None,rot_angle,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic2_120=Image_loader(path_pic2_120,color_channel,None,1,rot_angle,[y1_120,y2_120,x1_120,x2_120],None,None,[15,0],False,10/9).image
cam_pic2_240=Image_loader(path_pic2_240,color_channel,None,1,rot_angle,[y1_240,y2_240,x1_240,x2_240],None,None,[-15,0],False,10/9).image




#XRay measurement  2021-08-13 deltaz=202 shot 137

directory='pictures/2021-08-13/'
picture_name='shot_137.png'
picture2_name='shot_9.png' #background


#Define path for measured pictures
path_pic_front=directory+picture_name
path_pic_top=directory+picture_name
path_pic_120=directory+picture_name
path_pic_240=directory+picture_name

path_pic2_front=directory+picture2_name
path_pic2_top=directory+picture2_name
path_pic2_120=directory+picture2_name
path_pic2_240=directory+picture2_name




deltaz=202 #XRay measurements
s=0.0508
#s= 0.0634   #spatialresolution proton measurements
#deltaz=159# proton measurements
#X-Ray 2021-08-13 deltaz=208
shape_front=(210,180) #you can not reconstruct somethin bigger than 190
shape_side=(202,180)


rot_angle=180.4
#rot_angle=.4
x_0=int(50)
y_0=int(25)



y1_front=y_0+426
y2_front=y_0+636
x1_front=x_0+435
x2_front=x_0+615

y1_top=y_0+727
y2_top=y_0+929
x1_top=x_0+440
x2_top=x_0+620

y1_120=y_0+718
y2_120=y_0+920
x1_120=x_0+150
x2_120=x_0+330

y1_240=y_0+721
y2_240=y_0+923
x1_240=x_0+720
x2_240=x_0+900



###shift
shift_image_front =[0,0]
shift_image_top = [0,0]
shift_image_120 = [20,0]
shift_image_240 = [-16,0]




mask_border_front=[0,0,0,0]
mask_border_top=[0,0,0,0]
mask_border_120=[18,0,0,0]
mask_border_240=[0,18,0,0]





#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,rot_angle,[y1_front,y2_front,x1_front,x2_front],None,None,shift_image_front,mask_border_front,False,10/9).image  #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,rot_angle,[y1_top,y2_top,x1_top,x2_top],None,None,shift_image_top,mask_border_top,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,rot_angle,[y1_120,y2_120,x1_120,x2_120],None,None,shift_image_120,mask_border_120,False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,rot_angle,[y1_240,y2_240,x1_240,x2_240],None,None,shift_image_240,mask_border_240,False,10/9).image


cam_pic2_front=Image_loader(path_pic2_front,color_channel,None,None,rot_angle,[y1_front,y2_front,x1_front,x2_front],None,None,shift_image_front,mask_border_front,False,10/9).image
cam_pic2_top=Image_loader(path_pic2_top,color_channel,None,None,rot_angle,[y1_top,y2_top,x1_top,x2_top],None,None,shift_image_top ,mask_border_top,False,1).image
cam_pic2_120=Image_loader(path_pic2_120,color_channel,None,1,rot_angle,[y1_120,y2_120,x1_120,x2_120],None,None,shift_image_120,mask_border_120,False,10/9).image
cam_pic2_240=Image_loader(path_pic2_240,color_channel,None,1,rot_angle,[y1_240,y2_240,x1_240,x2_240],None,None,shift_image_240,mask_border_240,False,10/9).image










################################################################################################################################


#OncoRay measurement  2021-09-02 deltaz=140 shot 25
directory='pictures/2021-09-02/'
#directory='pictures/spatialresolution/'
picture_name='25.png'
picture2_name='background60s.png' #background


s= 0.073825   #spatialresolution proton measurements at oncoray
deltaz=140# proton measurements

rot_angle=89.7

x_0=int(0)
y_0=int(-5)

y1_front=y_0+450
y2_front=y_0+620
x1_front=x_0+458
x2_front=x_0+588

y1_top=y_0+657
y2_top=y_0+797
x1_top=x_0+459
x2_top=x_0+589

y1_120=y_0+656
y2_120=y_0+796
x1_120=x_0+267
x2_120=x_0+397

y1_240=y_0+658
y2_240=y_0+798
x1_240=x_0+649
x2_240=x_0+779

#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,rot_angle,[y1_front,y2_front,x1_front,x2_front],None,None,[0,-3],False,10/9).image     #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,rot_angle,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,rot_angle,[y1_120,y2_120,x1_120,x2_120],None,None,[7,0],False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,rot_angle,[y1_240,y2_240,x1_240,x2_240],None,None,[-10,0],False,10/9).image


cam_pic2_front=Image_loader(path_pic2_front,color_channel,None,None,rot_angle,[y1_front,y2_front,x1_front,x2_front],None,None,[0,-3],False,10/9).image
cam_pic2_top=Image_loader(path_pic2_top,color_channel,None,None,rot_angle,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic2_120=Image_loader(path_pic2_120,color_channel,None,1,rot_angle,[y1_120,y2_120,x1_120,x2_120],None,None,[7,0],False,10/9).image
cam_pic2_240=Image_loader(path_pic2_240,color_channel,None,1,rot_angle,[y1_240,y2_240,x1_240,x2_240],None,None,[-10,0],False,10/9).image




#OncoRay measurement  2021-09-02 deltaz=140 shot 5
directory='pictures/2021-09-02/'
picture_name='5.png'
picture2_name='background60s.png'

#Define path for measured pictures
path_pic_front=directory+picture_name
path_pic_top=directory+picture_name
path_pic_120=directory+picture_name
path_pic_240=directory+picture_name

path_pic2_front=directory+picture2_name
path_pic2_top=directory+picture2_name
path_pic2_120=directory+picture2_name
path_pic2_240=directory+picture2_name




#background
shape_front=(170,130) #you can not reconstruct somethin bigger than 190
shape_side=(140,130)


s= 0.073825   #spatialresolution proton measurements at oncoray
deltaz=140# proton measurements

rot_angle=89.9

x_0=int(0)
y_0=int(-5)

y1_front=y_0+451
y2_front=y_0+621
x1_front=x_0+457
x2_front=x_0+587

y1_top=y_0+657
y2_top=y_0+797
x1_top=x_0+459
x2_top=x_0+589

y1_120=y_0+657
y2_120=y_0+797
x1_120=x_0+267
x2_120=x_0+397

y1_240=y_0+658
y2_240=y_0+798
x1_240=x_0+649
x2_240=x_0+779

###shift
shift_image_front =[0,0]
shift_image_top = [0,0]
shift_image_120 = [8,0]
shift_image_240 = [-8,0]




mask_border_front=[0,0,0,0]
mask_border_top=[0,0,0,0]
mask_border_120=[10,0,0,0]
mask_border_240=[0,10,0,0]











###########shot7 and shot 9 and shot 15
directory='pictures/2021-09-02/'
picture_name='7.png'
picture2_name='background60s.png'

#Define path for measured pictures
path_pic_front=directory+picture_name
path_pic_top=directory+picture_name
path_pic_120=directory+picture_name
path_pic_240=directory+picture_name

path_pic2_front=directory+picture2_name
path_pic2_top=directory+picture2_name
path_pic2_120=directory+picture2_name
path_pic2_240=directory+picture2_name




#background
shape_front=(170,130) #you can not reconstruct somethin bigger than 190
shape_side=(140,130)


s= 0.073825   #spatialresolution proton measurements at oncoray
deltaz=140# proton measurements

rot_angle=89.9

x_0=int(0)
y_0=int(-5)

y1_front=y_0+451
y2_front=y_0+621
x1_front=x_0+457
x2_front=x_0+587

y1_top=y_0+657
y2_top=y_0+797
x1_top=x_0+459
x2_top=x_0+589

y1_120=y_0+657
y2_120=y_0+797
x1_120=x_0+267
x2_120=x_0+397

y1_240=y_0+658
y2_240=y_0+798
x1_240=x_0+649
x2_240=x_0+779


###shift
shift_image_front =[0,0]
shift_image_top = [0,0]
shift_image_120 = [8,0]
shift_image_240 = [-8,0]




mask_border_front=[0,0,0,0]
mask_border_top=[0,0,0,0]
mask_border_120=[10,0,0,0]
mask_border_240=[0,10,0,0]


################################n. 25 minibeams

background
shape_front=(170,130) #you can not reconstruct somethin bigger than 190
shape_side=(135,130)


s= 0.073825   #spatialresolution proton measurements at oncoray
deltaz=135# proton measurements

rot_angle=89.9

x_0=int(0)
y_0=int(-5)

y1_front=y_0+450
y2_front=y_0+620
x1_front=x_0+457
x2_front=x_0+587

y1_top=y_0+662
y2_top=y_0+797
x1_top=x_0+459
x2_top=x_0+589

y1_120=y_0+662
y2_120=y_0+797
x1_120=x_0+267
x2_120=x_0+397

y1_240=y_0+663
y2_240=y_0+798
x1_240=x_0+649
x2_240=x_0+779


###shift
shift_image_front =[0,0]
shift_image_top = [0,0]
shift_image_120 = [8,0]
shift_image_240 = [-10,0]




mask_border_front=[0,0,0,0]
mask_border_top=[0,10,0,0]
mask_border_120=[8,0,0,0]
mask_border_240=[0,8,0,0]















###############################################################################################################################



#OncoRay measurement  2021-09-01 deltaz=140

directory='pictures/2021-09-01/'
#directory='pictures/spatialresolution/'
picture_name='34.png'
picture2_name='background60s.png' #background

#Define path for measured pictures
path_pic_front=directory+picture_name
path_pic_top=directory+picture_name
path_pic_120=directory+picture_name
path_pic_240=directory+picture_name

path_pic2_front=directory+picture2_name
path_pic2_top=directory+picture2_name
path_pic2_120=directory+picture2_name
path_pic2_240=directory+picture2_name


s= 0.074   #spatialresolution proton measurements at oncoray
deltaz=140# proton measurements


rot_angle=89.9

x_0=int(0)
y_0=int(-5)

y1_front=y_0+450
y2_front=y_0+620
x1_front=x_0+458
x2_front=x_0+588

y1_top=y_0+657
y2_top=y_0+797
x1_top=x_0+459
x2_top=x_0+589

y1_120=y_0+652
y2_120=y_0+792
x1_120=x_0+267
x2_120=x_0+397

y1_240=y_0+652
y2_240=y_0+792
x1_240=x_0+649
x2_240=x_0+779



########################################################## n.1

rot_angle=89.7

x_0=int(0)
y_0=int(-5)

y1_front=y_0+450
y2_front=y_0+620
x1_front=x_0+458
x2_front=x_0+588

y1_top=y_0+657
y2_top=y_0+797
x1_top=x_0+459
x2_top=x_0+589

y1_120=y_0+651
y2_120=y_0+791
x1_120=x_0+265
x2_120=x_0+395

y1_240=y_0+654
y2_240=y_0+794
x1_240=x_0+654  #646
x2_240=x_0+784#776


###shift
shift_image_front =[0,0]
shift_image_top = [0,0]
shift_image_120 = [7,0]
shift_image_240 = [-10,0]




mask_border_front=[0,0,0,0]
mask_border_top=[0,0,0,0]
mask_border_120=[5,0,0,0]
mask_border_240=[0,5,0,0]


#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,rot_angle,[y1_front,y2_front,x1_front,x2_front],None,None,[0,-5],False,10/9).image     #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,rot_angle,[y1_top,y2_top,x1_top,x2_top],None,None,[0,0],False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,rot_angle,[y1_120,y2_120,x1_120,x2_120],None,None,[7,0],False,1000/752).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,rot_angle,[y1_240,y2_240,x1_240,x2_240],None,None,[-10,0],False,1000/752).image


cam_pic2_front=Image_loader(path_pic2_front,color_channel,None,None,rot_angle,[y1_front,y2_front,x1_front,x2_front],None,None,[0,-5],False,10/9).image
cam_pic2_top=Image_loader(path_pic2_top,color_channel,None,None,rot_angle,[y1_top,y2_top,x1_top,x2_top],None,None,None,False,1).image
cam_pic2_120=Image_loader(path_pic2_120,color_channel,None,1,rot_angle,[y1_120,y2_120,x1_120,x2_120],None,None,[7,0],False,1000/752).image
cam_pic2_240=Image_loader(path_pic2_240,color_channel,None,1,rot_angle,[y1_240,y2_240,x1_240,x2_240],None,None,[-10,0],False,1000/752).image



####################n.19

directory='pictures/2021-09-02/'
picture_name='19.png'
picture2_name='background60s.png'

#Define path for measured pictures
path_pic_front=directory+picture_name
path_pic_top=directory+picture_name
path_pic_120=directory+picture_name
path_pic_240=directory+picture_name

path_pic2_front=directory+picture2_name
path_pic2_top=directory+picture2_name
path_pic2_120=directory+picture2_name
path_pic2_240=directory+picture2_name




#background
shape_front=(170,130) #you can not reconstruct somethin bigger than 190
shape_side=(140,130)


s= 0.073825   #spatialresolution proton measurements at oncoray
deltaz=140# proton measurements

rot_angle=89.9

x_0=int(0)
y_0=int(-5)

y1_front=y_0+453
y2_front=y_0+623
x1_front=x_0+457
x2_front=x_0+587

y1_top=y_0+657
y2_top=y_0+797
x1_top=x_0+459
x2_top=x_0+589

y1_120=y_0+657
y2_120=y_0+797
x1_120=x_0+267
x2_120=x_0+397

y1_240=y_0+658
y2_240=y_0+798
x1_240=x_0+649
x2_240=x_0+779


###shift
shift_image_front =[0,0]
shift_image_top = [0,0]
shift_image_120 = [8,0]
shift_image_240 = [-10,0]




mask_border_front=[0,0,0,0]
mask_border_top=[0,0,0,0]
mask_border_120=[8,0,0,0]
mask_border_240=[0,8,0,0]
