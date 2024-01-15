import cv2

cap = cv2.VideoCapture(0)

#tracker = cv2.legacy.TrackerMOSSE_create()
tracker = cv2.TrackerCSRT_create()
success, img = cap.read()
bbox = cv2.selectROI("Tracking", img, False)
tracker.init(img, bbox)


# Dictionary to store IDs for each bounding box

id_dict = {}
object_id = 1

def drawbox(img, bbox, object_id):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x,y), ((x+w), (y+h)), (0,0,255), 3, 1)
    cv2.putText(img, f"Object Found {object_id}", (60, 80), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)


while True:
    timer = cv2.getTickCount()
    success, img = cap.read()

    success, bbox = tracker.update(img)

    if success:
        if object_id not in id_dict:
            id_dict[object_id] = bbox
        drawbox(img, bbox, object_id)
    else:
        cv2.putText(img, "Object Lost", (60, 80), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 190, 255), 2)


    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, str(int(fps)),(60,50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,140,120), 2)

    cv2.imshow("Tracking Object", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()