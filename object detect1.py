import os, sys
from os import listdir
import torch
from torchvision import transforms
from utils import *
import imageio
from PIL import Image, ImageDraw, ImageFont
#**********************************72***********************************
#allows for the use of GPU or switches back to cpu                      
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#loading our pretrained model from sgrvinod (https://github.com/sgrvinod)
checkpoint = 'C:/Users/sewmo/Desktop/Deep Learning and Computer Vision A-Z/Module 2 - Object Detection/checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint,map_location=torch.device('cpu'))
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch {}\n'.format(start_epoch))
model = checkpoint['model']
model = model.to(device)
model.eval()

#transforming image/frames so they will fit into the tensor
resize = transforms.Resize((300,300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229,0.224, 0.225])

#the method for detecting the objects
#uses the trained SSD300

""" 
   original : image or frame
   min_score: minimum threshold for a detected box to be considered a 
     match fo a certain class    
   max_overlap: maximum overlap two boxes 
     can have so that the one with the lower score is not suppressed 
   top_k: if there are a lot of detected objects across all the types 
     of objects keep only the top k
   suppress: which classes you don't want to include in object detection 
   return: the origanal image with the detected objects labeled

"""
def detect(original, min_score, max_overlap, top_k, suppress):
    
    # transform origanal image/frame
    image = normalize(to_tensor(resize(original)))

    # utilize the device we created
    image = image.to(device)

    #unsqueeze adds the "batch" layer to match the requirements of the tensor
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    #detect objects in SSD output 
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, 
                                                            predicted_scores, 
                                                            min_score = min_score, 
                                                            max_overlap = max_overlap, 
                                                            top_k = top_k)
    
    # sending detections to CPU
    det_boxes = det_boxes[0].to('cpu')

    # transforming the detected objects into original's dimensions
    original_dims = torch.FloatTensor([original_image.width, original_image.height,
                                       original_image.width, original_image.height]).unsqueeze(0)

    det_boxes = det_boxes * original_dims

    #get labels for detected objects
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    #when no ojects are found the  to ['0'] meaning background
    if det_labels == ['background']:
        # returns original image with no bounding boxes or labels
        return original_image
    
    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 30)
    
    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image

path =os.path.join(os.getcwd(), 'img_vid')
print (path)
video_path = os.path.join(path,'test.jpg')#put the video path in here
print(video_path)
#imageObjects = []
#imgFile = os.path.join( video_path, "funny_dog" )                    # Create full paths to images
        
    
original_image = Image.open(video_path, mode='r')
original_image = original_image.convert('RGB')
output_im = detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200,suppress= None).show()

import imageio
vpath =os.path.join(os.getcwd(), 'img_vid')
print (vpath)
file = "funny_dog.mp4"
video_path = os.path.join(path,file)#put the video path in here
reader = imageio.get_reader(video_path)

#getting the frames per second 
fpz = reader.get_meta_data()['fps']

#writing the new video with objects detected
writer = imageio.get_writer(file +'output.mp4', fps = fpz)
import numpy as np
# iterating through the video/images
for i, frame in enumerate(reader):
    original_image = Image.fromarray(frame)
    im = detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200,suppress= None)
    frame = np.array(im)
    writer.append_data(frame)
    print(i)
writer.close()