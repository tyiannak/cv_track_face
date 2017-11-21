import cv2
import numpy

(Width, Height) = (640, 480)

def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return numpy.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

cap = cv2.VideoCapture(0)
# width:
cap.set(3, Width)
# height:
cap.set(4, Height)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0

#glasses = cv2.imread("mask3.png", -1)

while(True):
    # Get current frame:
    ret, frame = cap.read()
    # Convert to gray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Dif = 0
    if count > 0:
        Dif = numpy.abs(gray.astype(float) - 
                        gray_prev.astype(float)).sum() / (128 * Height)

    count += 1
    # Apply face detector
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Show detected bounding boxes:
    for (x,y,w,h) in faces:
        x = x - w/6
        y = y - h/6
        w += w/6
        h += h/6
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #roi_color = blend_transparent(roi_color, cv2.resize(glasses, (h, w)))
        #frame[y:y + h, x:x + w] = roi_color

    if count > 1:
        for d in range(int(Dif)):
            cv2.putText(frame, "|" % Dif, (10+d, 10),
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255))

    cv2.imshow('win', frame)

    gray_prev = gray.copy()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
