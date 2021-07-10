import cv2
import pixellib
from pixellib.instance import instance_segmentation
from keras.models import load_model
from contours import *

current_dir = os.path.dirname(os.path.realpath('inference.py'))

result_prime = "Null"

# Function to extract frames
def FrameCapture(path,model):
      
    # Path to video file
    vidObj = cv2.VideoCapture(path)
    # to find the best plate from video
    pl_shape = (0,0,0)
    # Used as counter variable
    count = 0
    # checks whether frames were extracted
    success = 1
  
    while success:
  
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        if image is None:

            print("video completed!")
            break
        plate_img, plate, plate_rect = extract_plate(image)

        if plate == "None":

            print("No plate detected")
            continue
        else:

          print("Plate detected")
          cv2.imwrite(os.path.join(current_dir,"Output/licence_plate_p0.jpg"),plate)

          if count == 0:

              pl_shape = plate.shape
              char = segment_characters(plate)
              result = show_results(model,char)
              count +=1

              if len(result) in [9,10,11]:
                 
                  result_prime = result

          elif plate.shape > pl_shape:

            pl_shape = plate.shape
            cv2.imwrite(os.path.join(current_dir,"Output/licence_plate_p1.jpg"),plate)
            char = segment_characters(plate)
            result = show_results(model,char)

            if len(result) in [9,10,11]:
                cv2.imwrite(os.path.join(current_dir,"Output/licence_plate_p2.jpg"),plate)
                result_prime = result
                continue

            else:
              continue

        if len(plate_rect)> 1:
            continue
        
    return 0

if __name__ == "__main__":

    
    model = load_model(os.path.join(current_dir,"/Model/License_plate.hdf5"))
    call= FrameCapture('path_of_video_to_be_detected.mp4',model)
    print("The detected number plate is: ",result_prime)
