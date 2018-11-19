import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from PIL import Image
import sys

class Steganography(object):

    @staticmethod
    def __int_to_bin(rgb):
        r, g, b = rgb
        return ('{0:08b}'.format(r),
                '{0:08b}'.format(g),
                '{0:08b}'.format(b))

    @staticmethod
    def __bin_to_int(rgb):
        r, g, b = rgb
        return (int(r, 2),
                int(g, 2),
                int(b, 2))

    @staticmethod
    def __merge_rgb(rgb1, rgb2):
        r1, g1, b1 = rgb1
        r2, g2, b2 = rgb2
        
        rgb = (r1[:4] + r2[:4],
               g1[:4] + g2[:4],
               b1[:4] + b2[:4])
        return rgb

    @staticmethod
    def merge(img1, img2):


        if img2.size[0] > img1.size[0] or img2.size[1] > img1.size[1]:
            raise ValueError('Image 1 size is lower than image 2 size!')


        pixel_map1 = img1.load()
        pixel_map2 = img2.load()


        new_image = Image.new(img1.mode, img1.size)
        pixels_new = new_image.load()

        for i in range(img1.size[0]):
            for j in range(img1.size[1]):
                rgb1 = Steganography.__int_to_bin(pixel_map1[i, j])


                rgb2 = Steganography.__int_to_bin((0, 0, 0))

                if i < img2.size[0] and j < img2.size[1]:
                    rgb2 = Steganography.__int_to_bin(pixel_map2[i, j])

                rgb = Steganography.__merge_rgb(rgb1, rgb2)

                pixels_new[i, j] = Steganography.__bin_to_int(rgb)

        return new_image

    @staticmethod
    def unmerge(img):
        pixel_map = img.load()

        new_image = Image.new(img.mode, img.size)
        pixels_new = new_image.load()

        original_size = img.size

        for i in range(img.size[0]):
            for j in range(img.size[1]):
        
                r, g, b = Steganography.__int_to_bin(pixel_map[i, j])

                
                rgb = (r[4:] + "0000",
                       g[4:] + "0000",
                       b[4:] + "0000")

                pixels_new[i, j] = Steganography.__bin_to_int(rgb)

                if pixels_new[i, j] != (0, 0, 0):
                    original_size = (i + 1, j + 1)
        new_image = new_image.crop((0, 0, original_size[0], original_size[1]))

        return new_image


def steganography():
    print("Enter the path for image1")
    image1 = input()

    print("Enter the path for image2 or leave ir null")
    image2= input()

    print("Enter the path for output image")
    output = input()

    if image2 != "":
        img1 = Image.open(image1)
        img2 = Image.open(image2)

        merged_image = Steganography.merge(img1, img2)
        merged_image.save(output)

    else:
        img = Image.open(image1)

        unmerged_image = Steganography.unmerge(img)
        unmerged_image.save(output)

    
        
def basic_edge_detection():
    image = input("Enter the path of image")
    img = cv2.imread(image,0)
    plt.imshow(img)

    k=3
    t=137

    blur = cv2.GaussianBlur(img, (k, k), 0)
    (t, binary) = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)

    edgeX = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
    edgeY = cv2.Sobel(binary, cv2.CV_16S, 0, 1)

    edgeX = np.uint8(np.absolute(edgeX))
    edgeY = np.uint8(np.absolute(edgeY))
    edge = cv2.bitwise_or(edgeX, edgeY)

    plt.imshow(edge)        


def brightness():
    
    global image
    global height
    global width
    global channels
    
    name=input("Enter the name of image you want to edit: ")
    #print(name)
    image=cv2.imread(name)
    image=np.array(image)
    print(np.shape(image))
    channels=0
    height, width, channels=np.shape(image)
    print(channels)
    option=0  
    
    value=int(input("Enter the absolute amount by which you want to increase brightness: "))
    blue=image[:,:,0]
    green=image[:,:,1]
    red=image[:,:,2]

    if(channels==3):    
        blue=np.where((255-blue)<value,255,blue+value)
        green=np.where((255-green)<value,255,green+value)
        red=np.where((255-red)<value,255,red+value)
        

    blue=np.expand_dims(blue, axis=2)
    red=np.expand_dims(red, axis=2)
    green=np.expand_dims(green, axis=2)
    image2=np.concatenate([blue, green, red], axis=2)
    
    
    cv2.imshow('Image Before Editing',image)
    cv2.imshow('Image After Editing',image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contrast():
    global image
    global height
    global width
    global channels
    
    name=input("Enter the name of image you want to edit: ")
    #print(name)
    image=cv2.imread(name)
    image=np.array(image)
    print(np.shape(image))
    channels=0
    height, width, channels=np.shape(image)
    print(channels)
    option=0

    factor=float(input("Enter the factor by which you want to increase contrast: "))
    blue=image[:,:,0]
    green=image[:,:,1]
    red=image[:,:,2]
    
    cv2.imshow('Image Before Editing',image)

    if(channels==3):  
        for i in range(height):
            for j in range(width):
                if (blue[i][j] * factor > 255):
                    blue[i][j] = 255
                else:
                    blue[i][j] = blue[i][j] * factor
                    
                if (green[i][j] * factor > 255):
                    green[i][j] = 255
                else:
                    green[i][j] = green[i][j] * factor
                    
                if (red[i][j] * factor > 255):
                    red[i][j] = 255
                else:
                    red[i][j] = red[i][j] * factor        

    blue=np.expand_dims(blue, axis=2)
    red=np.expand_dims(red, axis=2)
    green=np.expand_dims(green, axis=2)
    image2=np.concatenate([blue, green, red], axis=2)

    cv2.imshow('Image After Editing',image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        

def greyscale():#0.21 R + 0.72 G + 0.07 B.
    
    global image
    global height
    global width
    global channels
    
    name=input("Enter the name of image you want to edit: ")
#print(name)
    image=cv2.imread(name)
    image=np.array(image)
    print(np.shape(image))
    channels=0
    height, width, channels=np.shape(image)
    print(channels)
    option=0
    
    

    if(channels!=3):
        print("Already in greyscale format")
    else:
        greyimage=image[:,:,0]*0.07+image[:,:,1]*0.72+image[:,:,2]*0.21
        greyimage=greyimage.astype('uint8')
        cv2.imshow('Image Before Editing',image)
        cv2.imshow('image',greyimage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
                   
def fusion():
    
    global image
    global height
    global width
    global channels
    
    name=input("Enter the name of image you want to edit: ")
#print(name)
    image=cv2.imread(name)
    image=np.array(image)
    print(np.shape(image))
    channels=0
    height, width, channels=np.shape(image)
    print(channels)
    option=0
    
    
    name=input("Enter the name of second image you want to fuse first image ")
    image2=cv2.imread(name)
    image2=np.array(image2)

    factor=float(input("Enter the fusion factor 0<factor<1(which will tell about dominance of second image over first image: "))

    while(factor<0 or factor>1):
        print("Invalid factor")
        factor=input("Enter the fusion factor 0<factor<1(which will tell about dominance of second image over first image: ")
    
    print(image[:, :, 0])
    cv2.imshow('Image Before Editing',image)
    
    blue1 = image[:, :, 0]
    green1 = image[:, :, 1]
    red1 = image[:, :, 2]
    
    blue2 = image2[:, :, 0]
    green2 = image2[:, :, 1]
    red2 = image2[:, :, 2]
    for i in range(height):
        for j in range(width):
            print(blue1[i][j])
            blue1[i][j] = int((1 - factor) * blue1[i][j])
            print(blue1[i][j])
            green1[i][j] = int((1 - factor) * green1[i][j])
            red1[i][j] = int((1 - factor) * red1[i][j])
            
            blue2[i][j]=int(factor * blue2[i][j])
            green2[i][j]=int(factor*green2[i][j])
            red2[i][j]=int(factor*red2[i][j])   


    for i in range(height):
        for j in range(width):
            if ((blue1[i][j] + blue2[i][j]) < 255):
                blue1[i][j] = blue1[i][j] + blue2[i][j]
            else:
                blue1[i][j] = 255

            if ((green1[i][j] + green2[i][j]) < 255):
                green1[i][j] = green1[i][j] + green2[i][j]
            else:
                green1[i][j] = 255

            if ((red1[i][j] + red2[i][j]) < 255) :
                red1[i][j] = red1[i][j] + red2[i][j]
            else:
                red1[i][j] = 255
    
    blue1=np.expand_dims(blue1, axis=2)
    red1=np.expand_dims(red1, axis=2)
    green1=np.expand_dims(green1, axis=2)
    image2=np.concatenate([blue1, green1, red1], axis=2)
    image2=image.astype('uint8')

    cv2.imshow('Image After Editing',image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def histogram():
    
    global image
    global channels
    global height
    global width
    
    name=input("Enter the name of image you want to edit: ")
#print(name)
    image=cv2.imread(name)
    image=np.array(image)
    print(np.shape(image))
    channels=0
    height, width, channels=np.shape(image)
    print(channels)
    option=0
    
    

    
    if(channels==3):
        image=image[:,:,0]*0.07+image[:,:,1]*0.72+image[:,:,2]*0.21
        image=image.astype('uint8')

    originalimage=image
    range1=int(input("Enter the range: "))
    array=np.reshape(image,(height*width))
    dictionary={}
    dictionary2={}
    
    unique, counts = np.unique(array, return_counts=True)
    dictionary=dict(zip(unique, counts))
    cumulative=0
    for key, value in dictionary.items():
        dictionary2[key]=value+cumulative
        cumulative+=value

    figure(num=None, figsize=(12,5), dpi=100, facecolor='w', edgecolor='k')
    plt.subplot(1, 2, 1)
    plt.bar(unique, counts)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency(Before Histogram Equilization')
    
    firstitem=int(dictionary2[next(iter(dictionary2))])
    print(firstitem)
    for key, value in dictionary2.items():
        dictionary2[key]=round((int(dictionary2[key])-firstitem)*(range1-1)/((height*width)-firstitem))
        #print(int(dictionary2[key])-firstitem)
        
    for i in range(height):
        for j in range(width):
            image[i,j]=dictionary2[image[i,j]]

    

    array2=np.reshape(image,(height*width))
    unique2, counts2 = np.unique(array, return_counts=True)
    
    plt.subplot(1, 2, 2)
    plt.bar(unique2, counts2)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency(After Histogram Equilization')

    plt.show()
    cv2.imshow('Image Before Editing',originalimage)
    cv2.imshow('Image After Editing',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def xyz():
    def fixedThresh():
        global img, blur, thresh, sel
        (t, mask) = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
        sel = cv2.bitwise_and(img, mask)
        cv2.imshow("image", sel)
    def adjustThresh(v):
        global thresh
        thresh = v
        fixedThresh()
    global filename,k,img,blur,thresh
    filename = "img2.jpg"
    k = 5
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(img, (k, k), 0)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    thresh = 128
    cv2.createTrackbar("thresh", "image", thresh, 255, adjustThresh)
    fixedThresh()
    k=cv2.waitKey(0)
    if(k==27):
        cv2.destroyAllWindows()
    elif(k==ord('s')):
        cv2.imwrite('mg.png',sel)
        cv2.destroyAllWindows()
        print("Image Saved")
    elif(k==ord('h')):
        equ=cv2.equalizeHist(sel)
        res = np.hstack((sel, equ)) 
        cv2.imshow('image', res) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 

def abc():
    def cannyEdge():
        global img, minT, maxT
        edge = cv2.Canny(img, minT, maxT)
        cv2.imshow("edges", edge)
    def adjustMinT(v):
        global minT
        minT = v
        cannyEdge()
    def adjustMaxT(v):
        global maxT
        maxT = v
        cannyEdge()
    global filename,img,maxT,minT
    filename = "img2.jpg"
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    minT = 30
    maxT = 150
    cv2.createTrackbar("minT", "edges", minT, 255, adjustMinT)
    cv2.createTrackbar("maxT", "edges", maxT, 255, adjustMaxT)
    cannyEdge()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def menu():
    global option
    print("1. Change Brightness")
    print("2. Change Contrast")
    print("3. Convert to Black White")
    print("4. Perform fusion of two images")
    print("5. Perform Histogram Equilization")
    print("6. Perform Steganography")
    print("7. Basic edge detection")
    print("8. Edge Detection with Threshold tracker")
    print("9. Edge Detection with trackbar")
    option=input("Enter the option number: ")
    if(option=='1'):
        brightness()
    elif(option=='2'):
        contrast()
    elif(option=='3'):
        greyscale()
    elif(option=='4'):
        fusion()
    elif(option=='5'):
        histogram()
    elif (option == '6'):
        steganography()
    elif (option == '7'):
        basic_edge_detection()
    elif (option == '8'):
        xyz()
    elif (option == '9'):
        abc()
    elif (option == 'q'):
        sys.exit()
        

menu()
print(option)
