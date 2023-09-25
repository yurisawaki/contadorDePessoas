import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import paho.mqtt.client as mqtt
import time

def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


MQTT_BROKER = "192.168.100.42"  
MQTT_PORT = 1883
MQTT_TOPIC = "contagem_de_pessoas"


pessoas = 0

# callback
def on_publish(client, userdata, mid):
    print("Mensagem publicada com sucesso")

# 
def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Tentando reconectar...")
        client.reconnect()

# Cliente MQTT
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.on_publish = on_publish
client.on_disconnect = on_disconnect  


camera = PiCamera()
camera.resolution = (640, 480)
raw_capture = PiRGBArray(camera, size=(640, 480))

fgbg = cv2.createBackgroundSubtractorMOG2()

posL = 150
offset = 20

xy1 = (0, posL)  

detects = []

total = 0
up = 0
down = 0

for frame in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
    image = frame.array

    xy2 = (image.shape[1], posL)  # Extremidade direita da tela (largura do quadro)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fgbmask = fgbg.apply(gray)

    retval, th = cv2.threshold(fgbmask, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)

    dilatation = cv2.dilate(opening, kernel, iterations=8)

    closing = cv2.morphologyEx(dilatation, cv2.MORPH_CLOSE, kernel, iterations=8)

    countours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    for cnt in countours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        detect = []

        if int(area) < 11000 or int(area) < 20000:
            centro = center(x, y, w, h)

            if len(detects) <= i:
                detects.append([])

            if centro[1] > posL - offset and centro[1] < posL + offset:
                detects[i].append(centro)
            else:
                detects[i].clear()

            i += 1

    if i == 0:
        detects.clear()

    if len(countours) == 0:
        for detect in detects:
            detects.clear()
    else:
        for detect in detects:
            if len(detect) > 0:
                for c in range(1, len(detect)):
                    if detect[c - 1][1] < posL and detect[c][1] > posL:
                        detect.clear()
                        up += 1
                        total += 1
                        continue

                    if detect[c - 1][1] > posL and detect[c][1] < posL:
                        detect.clear()
                        down += 1
                        total += 1
                        continue

    log_info = f"Total: {total}, Subindo: {up}, Descendo: {down}"
    print(log_info)

    # Publicar a contagem de pessoas 
    try:
        client.publish(MQTT_TOPIC, log_info)
    except (ConnectionError, OSError):
        print("Reconectando...")
        client.reconnect()

    raw_capture.truncate(0)
