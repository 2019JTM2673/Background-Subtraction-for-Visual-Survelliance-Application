########################### ML FOREGROUND AND BACKGROUND  SUBTRACTION FROM VEDIO ##############################################
################### import require libr.......................................##################################################
import cv2
import numpy as np
import math
height = 242
width = 352
A = []
sortin = []
new_wt = []
new_mean = []
new_variance = []
fr_pix = np.zeros([height, width])
mean1 = np.zeros([height, width])
mean2 = np.zeros([height, width])
mean3 = np.zeros([height, width])
mean4 = np.zeros([height, width])
variance1 = np.ones([height, width])
variance2 = np.ones([height, width])
variance3 = np.ones([height, width])
variance4 = np.ones([height, width])
weight1 = np.zeros([height, width])
weight2 = np.zeros([height, width])
weight3 = np.zeros([height, width])
weight4 = np.zeros([height, width])
mean = [0,0,0,0]
variance = [1,1,1,1]
weight = [0,0,0,0]
noofgauss = 4
alpha = 0.1

################################################## all function for Staffer grimson subtraction model###########################################################
def updategauss(pix, loca, rho):
    weight[loca] =  (1-alpha)*(weight[loca]) + alpha
    mean[loca] = ((1-rho)*mean[loca]) + (rho*pix)
    variance[loca] = ((1-rho)*variance[loca]) + (rho*(pix - mean[loca])*(pix - mean[loca]))
    # print(variance)
def search(pix):
    for i in range(noofgauss):
        if((pix-mean[i])/np.sqrt(variance[i]) <= 2.5):
            return i
    return -1

def new_search(pix,k):
    for i in range(k):
        if((pix-new_mean[i])/np.sqrt(new_variance[i]) <= 2.5):
            return i
    return -1

def prob(pix, loca):
    prt1 = 1/(np.sqrt(2*np.pi)*variance[loca])
    prt2 = np.power(((pix-mean[loca])/variance[loca]),2)
    prt3 = (np.exp(-prt2) * (prt1))
    return prt3

def createnewgauss(pix):
    flag=-1
    for i in range(noofgauss):
        if(weight[i] == 0):
            weight[i] = alpha
            flag = i
            ind = flag
            break
    if(flag==-1):
        ind = weight.index(min(weight))
    mean[ind] = pix
    variance[ind] = 10

            
        
###################################  opening the video file ##################################################################

videoread = cv2.VideoCapture('umcp.mpg')

if not videoread.isOpened():
	print ('File did not open')
	#return
count = 0
###########################################################################################################################################
###################################### START BREAKING VIDEO INTO FRAMES #######################################################################
while(videoread.isOpened()):
    count+=1
    ret, img1 = videoread.read()
    img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    row , col = img2.shape[:2]
    # print(row,col)
    if(count==800):
        break
#################################################  Start manipulation for each pixel of video frame ########################################
    for i in range (row):
        for j in range (col):
            
            mean = [mean1[i][j],mean2[i][j],mean3[i][j],mean4[i][j]]
            variance = [variance1[i][j], variance2[i][j],variance3[i][j],variance4[i][j]]
            weight = [weight1[i][j], weight2[i][j], weight3[i][j], weight4[i][j]]
            pix = img2[i][j]
            A = [weight1[i][j]/variance1[i][j],weight2[i][j]/variance2[i][j],weight3[i][j]/variance3[i][j],weight4[i][j]/variance4[i][j]]
            sortin = sorted(range(len(A)), key=lambda k: A[k])
            new_wt = [weight[sortin[3]], weight[sortin[2]], weight[sortin[1]], weight[sortin[0]]]
            new_mean = [mean[sortin[3]], mean[sortin[2]], mean[sortin[1]], mean[sortin[0]]]
            new_variance = [variance[sortin[3]], variance[sortin[2]], variance[sortin[1]], variance[sortin[0]]]
            sum = 0
            k = 0
            while(sum<0.8 and k<4):
                sum = sum + new_wt[k]
                k+=1
            # first k indices are the background gaussians
            new_loca = new_search(pix,k)
            if new_loca!= -1 :
                fr_pix[i][j] = 0
            else:
                fr_pix[i][j] = 255
            # if(i == 150 and j == 150):
            #     print(pix)

            # call the gaussian search function
############################################# Asignment of pixel value to each frame #####################################################################
            loca = search(pix)
            if loca == -1:
                createnewgauss(pix)
            else:
                rho = alpha*prob(pix, loca)
                updategauss(pix, loca, rho)
            wt = weight[1] + weight[2] + weight[3] + weight[0]
            for x in range(4):
                weight[x] = weight[x] / wt

            

            mean1[i][j] = mean[0]
            mean2[i][j] = mean[1]
            mean3[i][j] = mean[2]
            mean4[i][j] = mean[3]
            variance1[i][j] = variance[0]
            variance2[i][j] = variance[1]
            variance3[i][j] = variance[2]
            variance4[i][j] = variance[3]
            weight1[i][j] = weight[0]
            weight2[i][j] = weight[1]
            weight3[i][j] = weight[2]
            weight4[i][j] = weight[3]
            

    
    
    if(count == 5):
        
        print(mean1[150, 150])
        print(mean2[150, 150])
        print(mean3[150, 150])
        print(mean4[150, 150])
        print(weight1[150, 150])
        print(weight2[150, 150])
        print(weight3[150, 150])
        print(weight4[150, 150])
        print(variance1[150, 150])
        print(variance2[150, 150])
        print(variance3[150, 150])
        print(variance4[150, 150])
        print("\n")

    if(count == 150):
       
        print(mean1[150, 150])
        print(mean2[150, 150])
        print(mean3[150, 150])
        print(mean4[150, 150])
        print(weight1[150, 150])
        print(weight2[150, 150])
        print(weight3[150, 150])
        print(weight4[150, 150])
        print(variance1[150, 150])
        print(variance2[150, 150])
        print(variance3[150, 150])
        print(variance4[150, 150])
        



    cv2.imwrite("abc"+str(count)+".png", fr_pix)
videoread.release()
################################# END OF VIDEO #####################################################

# print("now the background starts")

# videoread = cv2.VideoCapture('umcp.mpg')

# if not videoread.isOpened():
# 	print ('File did not open')
# 	#return
# count = 0
# while(videoread.isOpened()):
#     count+=1
#     ret, img1 = videoread.read()
#     img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

#     for i in range (row):
#         for j in range (col):
#             pix = img2[i][j]
#             mean = [mean1[i][j],mean2[i][j],mean3[i][j],mean4[i][j]]
#             variance = [variance1[i][j], variance2[i][j],variance3[i][j],variance4[i][j]]
#             weight = [weight1[i][j], weight2[i][j], weight3[i][j], weight4[i][j]]
            
#             A = [weight1[i][j]/variance1[i][j],weight2[i][j]/variance2[i][j],weight3[i][j]/variance3[i][j],weight4[i][j]/variance4[i][j]]
#             sortin = sorted(range(len(A)), key=lambda k: A[k])
#             new_wt = [weight[sortin[3]], weight[sortin[2]], weight[sortin[1]], weight[sortin[0]]]
#             new_mean = [mean[sortin[3]], mean[sortin[2]], mean[sortin[1]], mean[sortin[0]]]
#             new_variance = [variance[sortin[3]], variance[sortin[2]], variance[sortin[1]], variance[sortin[0]]]
#             sum = 0
#             k = 0
#             while(sum<0.8):
#                 sum = sum + new_wt[k]
#                 k+=1
#             # first k indices are the background gaussians
#             new_loca = new_search(pix,k)
#             if new_loca!= -1 :
#                 fr_pix[i][j] = 0
#             else:
#                 fr_pix[i][j] = 255

#     cv2.imwrite("abc"+str(count)+".png", fr_pix)
