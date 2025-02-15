import cv2
import mediapipe as mp
import argparse

def process_image(img, face_detection, shapes):
    H = shapes[0]
    W = shapes[1]
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
    return img

# set mode and path
mode = 'camera'
path = {
    'image': "tom_brady.png",
    'video': "video.mp4",
    'camera': ""
}
parser = argparse.ArgumentParser()
parser.add_argument("--mode", default=mode)
parser.add_argument("--filePath", default=path[mode])
args = parser.parse_args()
print(f"\nMode: {args.mode}")
print(f"File Path: {args.filePath}\n")

# detect faces
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

if args.mode in ["image"]:
    # read image
    img = cv2.imread(args.filePath)
    H, W, _ = img.shape
    img = process_image(img, face_detection, (H, W)) 
    cv2.imwrite("out.png", img)

elif args.mode == "video":
    video = cv2.VideoCapture(args.filePath)

    if not video.isOpened():
        print("Error opening video")
        exit()

    ret, frame = video.read()
    if not ret:
        print("Error: Couldn't read the first frame.")
        exit()

    H, W, _ = frame.shape

    output_video = cv2.VideoWriter(
        "out.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),  
        25,
        (W, H)
    )

    while True:
        ret, frame = video.read()
        if not ret:
            print("No more frames to read or error occurred.")
            break

        if frame is None or frame.size == 0:
            print("Received an empty frame, skipping...")
            continue

        img = process_image(frame, face_detection, (H, W))
        
        cv2.imshow("Blurred Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

        output_video.write(img)

    video.release()
    output_video.release()
    cv2.destroyAllWindows()

elif args.mode in ["camera"]:
    # open video
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error opening camera")
        exit() 

    while True:
        ret, frame = webcam.read()
        H, W, _ = frame.shape
        img = process_image(frame, face_detection, (H, W)) 
        cv2.imshow('frame', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    webcam.release() 
    cv2.destroyAllWindows()

face_detection.close()