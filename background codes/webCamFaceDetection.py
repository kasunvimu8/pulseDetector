import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
vedio_capture = cv.VideoCapture(0)

while True:

    # capture frame by frame, ret have the return code.usefull when reading from a file vedio,
    # tell run out of frames
    ret, frame = vedio_capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv.imshow('vedio', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
vedio_capture.release()
cv.destroyAllWindows()
