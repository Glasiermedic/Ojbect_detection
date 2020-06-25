# library imports
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

#defining a function for producton 
def detect(frame, net, transform):
    height, width = frame.shape[:2]

    # transform frame into a numpy array
    # transform returns two elements we just want the first one
    frame_t = transform(frame)[0] #numpy-array
    #converts numpy to torch tensor
    x = torch.from_numpy(frame_t).permute(2,0,1)
    x = Variable(x.unsqueeze(0))
    #feeding images into neural network
    y = net(x)
    #getting the tensor of the data we want
    detections = y.data
    # detections = [batch, number of the type of object,  the number of times an object occurs
    # (score,x0,y0,x1,y1)]

    #create new tensor with coordinates (x1,y1,x2,y2)
    #which correspond with the upper left and bottom right corners
    scale = torch.Tensor([width, height, width, height])
    
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6:
            pt = (detections[0,i,j,1:]*scale).numpy()
            cv2.rectangle(frame, (int(pt[0]),int(pt[1])),(int(pt[2]),int(pt[3])),(255,0,0),3)
            cv2.putText(frame, labelmap[i-1],(int(pt[0]),int(pt[1])),cv2.FONT_HERSHEY_SIMPLEX,(255,255,0),2,cv2.LINE_AA)
            j +=1

    return frame

#create ssd neaural network 
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',map_location = lambda storage, loc: storage))

#creating the transform to make the frames compatible with the neural network
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

import os
path =os.path.join(os.getcwd(), 'img_vid')
print (path)
video_path = os.path.join(path,'funny_dog.mp4')#put the video path in here
reader = imageio.get_reader(video_path)

#getting the frames per second 
fpz = reader.get_meta_data()['fps']

#writing the new video with objects detected
writer = imageio.get_writer('output.mp4', fps = fpz)

# iterating through the video/images
for i, frame in enumerate(reader):
   frame = detect(frame,net.eval(),transform)
   writer.append_data(frame)
   print(i)
writer.close()




