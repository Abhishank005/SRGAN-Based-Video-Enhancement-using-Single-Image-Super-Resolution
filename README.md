# SRGAN Based Video-Enhancement-using-Single-Image-Super-Resolution

In this project, we have used a pretrained Super-Resolution Generative Adversarial Networks(SRGAN) model to perform Video enhancement using Single Image Super Resolution. The model takes a low resolution video as input and provides a high resolution video as output. The model has been validated in different genres videos at different quality levels. It is done in 6 simple steps which includes conversion of input video into low resolution frames and then converting back the processed high resolution frames into the output video.
## STEP-1
Clone the following repository which consists of Pretrained SRGAN Model in your Notebook:-

``` !git clone https://github.com/krasserm/super-resolution ```

## STEP-2
Create a directory named Super Resolution by using the command given below :-

```cd /content/super-resolution```

## STEP-3
The pretrained weights required for running the model can be  downloaded from the link given below:-
[weights-srgan.tar.gz](https://drive.google.com/file/d/1ZKpQvtxLKKq2fM1gKtl085pgHSgSQSBw/view?usp=sharing)

After downloading the pretrained weights, upload it in your notebook and then run the command below to extract the weights into the root folder-
```!tar xvfz /content/weights-srgan.tar.gz```


## STEP-4
To perform video enhancement,the input video should be converted into frames and the model can be used to obtain super resolved frames.This can be done using python codes given below.

```python
# Importing all necessary libraries 
import timei
import cv2 
import os
import numpy as np
from model import resolve_single
from utils import load_image, plot_sample
from model.srgan import generator

# Read the video from specified path 
cam = cv2.VideoCapture("/content/Drama144p_input.3gp") 
fps = cam.get(cv2.CAP_PROP_FPS)
print(fps)


try:
      
    # creating a folder named data 
    if not os.path.exists('data'): 
        os.makedirs('data') 
  
# if not created then raise error 
except OSError:
    print ('Error: Creating directory of data') 
  
#frames Extraction from video 
currentframe = 0
arr_img = []
while(True): 
      
    # reading from frame 
    ret,frame = cam.read() 
  
    if ret: 
        # if video is still left continue creating images 
        name = './data/frame' + str(currentframe).zfill(3) + '.jpg'
        print ('Creating...' + name) 
  
        # writing the extracted images 
        cv2.imwrite(name, frame) 
  
        # increasing counter so that it will show how many frames are created 
        currentframe += 1
        #storing the path of extracted frames in a list
        arr_img.append(name)
    else: 
        break
#print(arr_img)

start = timeit.default_timer()
model = generator()
model.load_weights('weights/srgan/gan_generator.h5')

#Initialization of an empty list to store the super resolved images
arr_output=[]
print(len(arr_img))
n= len(arr_img)

#Implementation of SRGAN Model in extracted frames
for i in range(n):
  lr = load_image(arr_img[i])
  sr = resolve_single(model, lr)
  #plot_sample(lr, sr)
  
  arr_output.append(sr)
stop = timeit.default_timer()
#print(arr_output)

print("time : ", stop-start)

# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows()
```

Here we have attatched a sample image that shows model implementation on an input frame-

![Results](Results/Results.png)

# STEP-5
Run the Python codes given below to save the super resolved frames obtained in a folder and to store their output path in a list-

```python
#Importing necessary libraries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img

#Making a directory for storing super resolved frames in image format
os.makedirs("output_images")

#Initialization of an empty list to store the path of Super resolved frames
s_res= []
for j in range(len(arr_output)):
  out_name = '/content/super-resolution/output_images/frame' + str(j).zfill(3) + '.jpg'
  img_pil = array_to_img(arr_output[j])
  img1 = save_img(out_name, img_pil)
  s_res.append(out_name)
  
#print(s_res)
```

# STEP-6
Run the Python codes given  below to  convert  super resolved frames obtained into a Video-

```python
import cv2
import numpy as np
for i in range(len(s_res)):
    filename=s_res[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)

fps = 20       #Put the fps value as your convenience or 
               #Calculate by using (No. of frames)/Video_duration in seconds  

#Creation of output video               
out = cv2.VideoWriter('drama2_output.mp4',cv2.VideoWriter_fourcc(*'DIVX'), fps , size)

#Writing Frames into video
for i in range(len(s_res)):
    out.write(cv2.imread(s_res[i]))
out.release()
```

# Final Results - 
Below link provides the results obtained for Video enhancement using single image super resolution using SRGAN Model. 

[Video Results](https://drive.google.com/drive/folders/1NiyJCLsB_-pAmFJNF97QhZiho7zPLMCw?usp=sharing)


