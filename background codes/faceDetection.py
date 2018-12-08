import cv2 as cv

def  detect_faces (face_cascade ,colored_image ,sacleFactor):
    img_copy = colored_image.copy()
    #convert the test image to gray image as opencv face detector expects gray images
    gray =cv.cvtColor(img_copy,cv.COLOR_BGR2GRAY)
    faces =face_cascade.detectMultiScale(gray, sacleFactor, 5)
    #print the number of faces found
    #print('Faces found: ', len(faces))
    #go over list of faces and draw them as rectangles on original colored
    for (x,y,w,h) in faces :
        cv.rectangle(img_copy,(x,y),(x+w ,y+h),(255,0,0),2)

    return img_copy


face_cascade =cv.CascadeClassifier('haarcascade_frontalface_default.xml')
img =cv.imread('sanga.png')
detect_faces(face_cascade,img,1.3)
cv.imshow('image',detect_faces(face_cascade,img,1.3))
cv.waitKey(0)
cv.destroyAllWindows()
