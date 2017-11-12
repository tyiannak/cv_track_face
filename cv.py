import cv2
import numpy

(Width, Height) = (640, 480)

def plotCV(Fun, Width, Height, MAX):
    if len(Fun)>Width:
        hist_item = Height * (Fun[len(Fun)-Width-1:-1] / MAX)
    else:
        hist_item = Height * (Fun / MAX)
    h = numpy.zeros((Height, Width, 3))
    hist = numpy.int32(numpy.around(hist_item))

    for x,y in enumerate(hist):
        cv2.line(h,(x,Height),(x,Height-y),(255,0,255))
    return h

cap = cv2.VideoCapture(0)
# width:
cap.set(3, Width)
# height:
cap.set(4, Height)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
while(True):
    # Get current frame:
    ret, frame = cap.read()
    # Convert to gray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Dif = 0
    if count > 0:
        Dif = numpy.abs(gray.astype(float) - grayPrev.astype(float)).sum() / (128 * Height)

    count += 1
    # Apply face detector
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Show detected bounding boxes:
    for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = frame[y:y+h, x:x+w]

    if count > 1:
        for d in range(int(Dif)):
            cv2.putText(frame, "|" % Dif, (10+d, 10),
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255))

    cv2.imshow('win', frame)

    grayPrev = gray.copy()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #h = plotCV(numpy.repeat(histRGBratio, WidthPlot2 / histRGBratio.shape[0]),
    #           WidthPlot2, Height, numpy.max(histRGBratio));
    #cv2.imshow('frame', h)

cap.release()
cv2.destroyAllWindows()
