from flet import *
import numpy as np
import cv2
import base64
import time
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
import io

def main(page: Page):

    myresult = Column()
    record_list = DataTable(
        columns=[
            DataColumn(Text("Object")),
            DataColumn(Text("Confidence")),
            DataColumn(Text("Points")),
            DataColumn(Text("Timestamp")),
        ],
        rows=[]
    )

    # Función para cargar registros desde JSON
    def load_records_from_json():
        with open('records.json', 'r') as file:
            return json.load(file)

    # Cargar registros iniciales desde el archivo JSON
    object_records = load_records_from_json()
    
    # Añadir registros iniciales a la tabla
    for record in object_records:
        label, confidence, points, timestamp = record['label'], record['confidence'], record['points'], record['timestamp']
        record_list.rows.append(
            DataRow(
                cells=[
                    DataCell(Text(label)),
                    DataCell(Text(f"{confidence:.2f}")),
                    DataCell(Text(f"{points}")),
                    DataCell(Text(timestamp)),
                ]
            )
        )

    # Load YOLO model
    model = YOLO('best.pt')  # Load a pretrained YOLOv8 model from ultralytics

    # Points mapping for different objects
    points_mapping = {
        "PLASTIC": 5,
        "CARDBOARD": 2,
        "BIODEGRADABLE": 5,
        "GLASS": 5,
        "METAL": 5,
        "PAPER": 5
    }

    # Create a black image as a placeholder
    black_image = np.zeros((480, 640, 3), np.uint8)
    _, buffer = cv2.imencode('.jpg', black_image)
    black_image_str = base64.b64encode(buffer).decode()
    image_control = Image(src_base64=black_image_str)
    detection_text = Text("Detected: None", size=25, weight="bold")

    detected_objects = {}
    freeze_frame = False
    freeze_time = 3  # seconds

    def detect_objects(e):
        nonlocal freeze_frame
        nonlocal detected_objects

        # Reset the freeze_frame flag and detected_objects dictionary
        freeze_frame = False
        detected_objects = {}
        detection_text.value = "Detected: None"
        page.update()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        while True:
            if freeze_frame:
                # Display the last frozen frame
                page.update()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            results = model(frame)

            current_time = time.time()
            new_detected_objects = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label = model.names[int(box.cls)]
                    confidence = float(box.conf)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    new_detected_objects.append(label)

                    if label in detected_objects:
                        if current_time - detected_objects[label] > freeze_time:
                            freeze_frame = True
                            points = points_mapping.get(label, 0)
                            detection_text.value = f"Freezed on: {label} ({confidence:.2f})"
                            object_records.append({"label": label, "confidence": confidence, "points": points, "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))})
                            update_record_list()
                            page.update()
                            # Insert the detected object into the JSON file
                            insert_record_into_json({"label": label, "confidence": confidence, "points": points, "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))})
                    else:
                        detected_objects[label] = current_time

            # Remove old detected objects that are no longer detected
            for label in list(detected_objects.keys()):
                if label not in new_detected_objects:
                    del detected_objects[label]

            if not freeze_frame:
                if new_detected_objects:
                    detection_text.value = "Detected: " + ", ".join(new_detected_objects)
                    page.update()

                # Encode frame to base64 to display in Flet app
                _, buffer = cv2.imencode('.jpg', frame)
                img_str = base64.b64encode(buffer).decode()
                image_control.src_base64 = img_str
                page.update()

            # If you press 'q' in the webcam window then close the webcam
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if cap.isOpened():
            cap.release()

    def insert_record_into_json(record):
        with open('records.json', 'r+') as file:
            data = json.load(file)
            data.append(record)
            file.seek(0)
            json.dump(data, file, indent=4)

    def update_record_list():
        record_list.rows.clear()
        for record in object_records:
            label, confidence, points, timestamp = record['label'], record['confidence'], record['points'], record['timestamp']
            record_list.rows.append(
                DataRow(
                    cells=[
                        DataCell(Text(label)),
                        DataCell(Text(f"{confidence:.2f}")),
                        DataCell(Text(f"{points}")),
                        DataCell(Text(timestamp)),
                    ]
                )
            )
        page.update()

    # New Home Tab Content
    home_image_path = "imagenes/contenedores.jpg"  # Update with the path to your image
    with open(home_image_path, "rb") as image_file:
        home_image_str = base64.b64encode(image_file.read()).decode()

    home_content = Column([
        Text("Welcome to the Object Detection App", size=30, weight="bold"),
        Image(src_base64=home_image_str, width=640, height=480)
    ])

    # Generate historical points and charts
    def generate_historical_data():
        month_points = defaultdict(int)
        for record in object_records:
            month = record['timestamp'][:7]  # Extraer el año y el mes (YYYY-MM)
            month_points[month] += record['points']

        months = sorted(month_points.keys())
        points = [month_points[month] for month in months]
        
        return months, points

    def create_chart(months, points):
        plt.switch_backend('Agg')  # Use non-GUI backend
        plt.figure(figsize=(10, 5))
        plt.plot(months, points, marker='o', linestyle='-', color='b')
        plt.xlabel('Month')
        plt.ylabel('Points')
        plt.title('Accumulated Points Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save chart to a string in base64 format
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        
        return img_str

    months, points = generate_historical_data()
    chart_image_str = create_chart(months, points)
    chart_image = Image(src_base64=chart_image_str)

    historical_content = Column([
        Text("Historical Points", size=30, weight="bold"),
        chart_image
    ])

    tab_control = Tabs(
        tabs=[
            Tab(text="Home", content=home_content),
            Tab(text="Detector", content=Column([
                Text("Object Detector", size=30, weight="bold"),
                ElevatedButton("Open Camera",
                            bgcolor="blue", color="white",
                            on_click=detect_objects
                ),
                # Show result from your YOLO detection to text widget
                Text("Detected Objects:", size=20, weight="bold"),
                Divider(),
                myresult,
                detection_text,
                image_control
            ])),
            Tab(text="Records", content=Column([
                Text("Detected Object Records", size=30, weight="bold"),
                record_list
            ])),
            Tab(text="Historical Data", content=historical_content)
        ]
    )

    page.add(tab_control)

app(target=main)
