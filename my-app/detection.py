import cv2
import time
from ultralytics import YOLO
import json
from utils import update_record_list
import base64
import numpy as np
from threading import Thread

# Load YOLO model
model = YOLO('besto.pt')

# Mappings for labels, points, containers, and instructions
points_mapping = {
    "Plástico": 30,
    "Cartón": 20,
    "Biodegradable": 15,
    "Vidrio": 25,
    "Metal": 30,
    "Papel": 20
}

names_esp_mapping = {
    "PLASTIC": "Plástico",
    "CARDBOARD": "Cartón",
    "BIODEGRADABLE": "Biodegradable",
    "GLASS": "Vidrio",
    "METAL": "Metal",
    "PAPER": "Papel"
}

container_mapping = {
    "PLASTIC": "amarillo",
    "CARDBOARD": "azul",
    "BIODEGRADABLE": "marrón",
    "GLASS": "verde",
    "METAL": "amarillo",
    "PAPER": "azul"
}

instructions_mapping = {
    "Plástico": "Deposítalo en el contenedor amarillo y verifica que esté limpio. Reciclar plástico ayuda a reducir la contaminación y a conservar los recursos naturales.",
    "Cartón": "Deposítalo en el contenedor azul y verifica que esté limpio. Si no, ponlo en el orgánico. Reciclar cartón reduce la deforestación y disminuye la cantidad de residuos en los vertederos.",
    "Biodegradable": "Deposítalo en el contenedor marrón. Compostar residuos orgánicos enriquece el suelo y reduce la producción de gases de efecto invernadero.",
    "Vidrio": "Deposítalo en el contenedor verde y verifica que no tengas tapas de otro material que deberia ir en el contenedor amarillo. Reciclar vidrio ahorra energía y disminuye la necesidad de extraer nuevas materias primas.",
    "Metal": "Deposítalo en el contenedor amarillo. Reciclar metales ayuda a conservar recursos naturales y reduce la contaminación del aire y el agua.",
    "Papel": "Deposítalo en el contenedor azul, si esta sucio depositalo en el organico. Reciclar papel reduce la tala de árboles y disminuye el impacto ambiental de la producción de papel nuevo."
}

# Global variables
camera_running = False
inference_running = False
freeze_frame = False
last_frame = None
frame_count_threshold = 10
detection_counts = {}
frame_skip = 5  # Number of frames to skip

def create_black_background(width=420, height=420):
    """Create a black background with specified dimensions."""
    return np.zeros((height, width, 3), dtype=np.uint8)

def set_image_control(image_control, image=None, width=420, height=420):
    """Set the image for the control with optional resizing."""
    if image is None:
        image = create_black_background(width, height)
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height))
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode()
    image_control.src_base64 = img_str
    image_control.update()

def clear_display(image_control):
    """Clear the display by setting a black background."""
    set_image_control(image_control, create_black_background())

def initialize_detection():
    """Initialize detection variables."""
    global camera_running, inference_running, freeze_frame, last_frame, detection_counts
    camera_running = False
    inference_running = True
    freeze_frame = False
    last_frame = None
    detection_counts = {}

def cleanup_detection(cap):
    """Cleanup resources after detection."""
    global inference_running
    if cap.isOpened():
        cap.release()
    inference_running = False

def detect_objects(instruction_label, detection_text, image_control, object_records, record_list, page):
    """Detect objects and update display accordingly."""
    global camera_running, inference_running, freeze_frame, last_frame, detection_counts

    initialize_detection()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        cleanup_detection(cap)
        return

    width, height = 420, 420
    clear_display(image_control)
    detection_text.value = ""
    instruction_label.value = "Instrucciónes: None"
    page.update()

    frame_count = 0

    while inference_running:
        if freeze_frame:
            if last_frame is not None:
                set_image_control(image_control, last_frame, width, height)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (width, height))
        results = model(frame)
        detected_objects = []
        current_detected_labels = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detected_label = names_esp_mapping.get(model.names[int(box.cls)], "desconocido")
                confidence = float(box.conf)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(frame, f"{detected_label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                detected_objects.append(detected_label)
                current_detected_labels.append(detected_label)

        for detected_label in current_detected_labels:
            if detected_label in detection_counts:
                detection_counts[detected_label] += 1
                if detection_counts[detected_label] >= frame_count_threshold:
                    freeze_frame = True
                    last_frame = frame.copy()
                    points = points_mapping.get(detected_label, 0)
                    container = container_mapping.get(detected_label, "desconocido")
                    instruction = instructions_mapping.get(detected_label, "Instrucción no disponible.")
                    detection_text.value = f"Objeto detectado: {detected_label} ({confidence:.2f})"
                    instruction_label.value = f"Instrucciónes: {instruction}"
                    object_record = {
                        "label": detected_label,
                        "confidence": confidence,
                        "points": points,
                        "container": container,
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    }
                    object_records.append(object_record)
                    update_record_list(record_list, object_records, page)
                    page.update()
                    insert_record_into_json(object_record)

                    cleanup_detection(cap)
                    return

            else:
                detection_counts[detected_label] = 1

        for detected_label in list(detection_counts.keys()):
            if detected_label not in current_detected_labels:
                detection_counts[detected_label] = 0

        if not freeze_frame:
            if detected_objects:
                detection_text.value = "Parece que es: " + ", ".join(detected_objects)
                page.update()

            set_image_control(image_control, frame, width, height)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cleanup_detection(cap)

def insert_record_into_json(record):
    """Insert a record into a JSON file."""
    try:
        try:
            with open('records.json', 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
        except FileNotFoundError:
            data = []

        data.append(record)

        with open('records.json', 'w') as file:
            json.dump(data, file, indent=4)

        print("Record successfully saved.")

    except Exception as e:
        print(f"Error saving record: {e}")

def camera(image_control, page):
    """Start camera feed and display images."""
    global camera_running
    camera_running = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        camera_running = False
        return

    width, height = 420, 420
    set_image_control(image_control, create_black_background(width, height))
    page.update()

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        frame = cv2.resize(frame, (width, height))
        set_image_control(image_control, frame, width, height)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera_running = False
            break

    if cap.isOpened():
        cap.release()
    clear_display(image_control)

def on_button_press(instruction_label, detection_text, image_control, object_records, record_list, page):
    """Handle button press to start object detection."""
    global camera_running, freeze_frame, detection_counts, inference_running
    # Stop any ongoing processes
    camera_running = False
    freeze_frame = False
    last_frame = None
    detection_counts = {}
    inference_running = True  # Start new inference

    # Clear the image display
    clear_display(image_control)
    detection_text.value = ""
    instruction_label.value = "Instrucciónes: None"
    page.update()

    # Start object detection in a separate thread to avoid blocking the main thread
    detection_thread = Thread(target=detect_objects, args=(instruction_label, detection_text, image_control, object_records, record_list, page))
    detection_thread.start()
