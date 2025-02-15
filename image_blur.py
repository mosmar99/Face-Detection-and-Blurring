import cv2
import mediapipe as mp

# read image
img_path = 'tom_brady.png'
img = cv2.imread(img_path)
H, W, _ = img.shape

# detect faces
# model selection 0 (faces are within 2 meter from the camera)
# face detector assumes RBG colorspace
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
out = face_detection.process(img_rgb)

# if there at least one human face detected
if out.detections != None:
    for detection in out.detections:
        location_data = detection.location_data
        bbox = location_data.relative_bounding_box

        # returns perc. of width and height
        x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
        
        x1 = int(x1 * W)
        y1 = int(y1 * H)
        w = int(w * W)
        h = int(h * H)

        # blur faces
        k_size = 42
        img[y1:y1+h, x1:x1+w] = cv2.blur(img[y1:y1+h, x1:x1+w], (k_size,k_size))

    cv2.imshow('img', img)
    cv2.waitKey(0)
    face_detection.close()