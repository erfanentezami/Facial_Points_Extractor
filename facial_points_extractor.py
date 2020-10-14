import dlib
import cv2
import glob

# for inputing your path remember to change all \ to /      
model = "shape_predictor_68_face_landmarks.dat"
path="============ input your images path ============"

frontalFaceDetector = dlib.get_frontal_face_detector()

faceLandmarkDetector = dlib.shape_predictor(model)


# loading images
images_path = glob.glob(path + "*.png") + glob.glob(path + "*.jpg")
images_path.sort()

im_start_point=len(path)

# images counter
img_num=1

# number of your images
max_sample = len(images_path)

total_images = []
images_name = []

# reading images using openCV
with open("images_info.txt", 'w') as f:
 for im in images_path :
    image = cv2.imread(im)
    images_name.append(str(im[im_start_point:]))
    total_images.append(image)
    shape=str(image.shape)
    f.write("IMAGE NUMBER %i   NAME : %s   SHAPE : %s\n"%(img_num,str(im[im_start_point:]),shape))
    img_num += 1
f.close()

# reset the images counter
img_num=1

print("your total_images list size is {}".format(total_images.__len__()))

with open("RESULT.csv", 'w') as f:
 f.write("name,number,face,")  
 for i in range(1,69):
     f.write("%sx,%sy,"%(int(i),int(i)))
 # to use this code for ML you can put your labels in the last column
 f.write("label,")
 f.write("\n")

 for i in range(max_sample):
  
   faces = frontalFaceDetector(total_images[i], 0)
   print("List of all faces detected in image %i : "%int(img_num),len(faces))
   
   # detecting all faces one by one and apply landmarks on them
   for k in range(0, len(faces)):
    # dlib rectangle class will detecting face so that landmark can apply inside of that area
      faceRectangleDlib = dlib.rectangle(int(faces[k].left()),int(faces[k].top()),
         int(faces[k].right()),int(faces[k].bottom()))

      # Now we are running loop on every detected face and putting landmark on that with the help of faceLandmarkDetector
      detectedLandmarks = faceLandmarkDetector(total_images[i], faceRectangleDlib)

       # count number of landmarks we actually detected on image
      if k==0:
         land_num = len(detectedLandmarks.parts())
         print("Total number of face landmarks detected : ", land_num)   
      
      # Write total result to csv file
      # img_num is your image number and k is the number of extra detected faces start from 0
      f.write("%s,%s,%s,"%(str(images_name[i]),int(img_num),int(k)))
      for p in detectedLandmarks.parts():
          f.write("%s,%s," %(int(p.x),int(p.y)))
      f.write("\n")

   img_num += 1
   
 f.close()   
