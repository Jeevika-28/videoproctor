import cv2
import boto3
import os
from ultralytics import YOLO

s3 = boto3.client('s3')
model = YOLO("bestt.pt")

def lambda_handler(event, context):
    for record in event['Records']:
        bucket_name = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']
        

        download_path = f'/tmp/{bucket_name}_{os.path.basename(object_key)}'
        
        s3.download_file(bucket_name, object_key, download_path)
        
        cap = cv2.VideoCapture(download_path)

        count = 0
        skip = 30

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            count += 1

            if count % skip == 0:
                results = model.predict(frame, show=False)
                result = results[0]

                if len(result) > 1:
                    for box in result.boxes:
                        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        filename = f"/tmp/frame_{count}.jpg"
                        cv2.imwrite(filename, frame)
                        upload_key = f"processed_frames/frame_{count}.jpg"
                        s3.upload_file(filename, 'studentproctordata', upload_key)

        cap.release()
        cv2.destroyAllWindows()