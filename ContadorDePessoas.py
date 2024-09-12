import cv2
import paho.mqtt.client as mqtt

def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Configurar informações de conexão MQTT
MQTT_BROKER = "broker.emqx.io"  
MQTT_PORT = 1883
MQTT_TOPIC1 = "/UFPA/LAAI/contagem_de_pessoas/entrada/"
MQTT_TOPIC2 = "/UFPA/LAAI/contagem_de_pessoas/saida/"

# Variáveis para contar as pessoas
pessoas = 0

def on_publish(client, userdata, mid):
   print("mensagem publicada")

# Função de callback quando o cliente MQTT é desconectado
def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Desconexão inesperada. Tentando reconectar...")
        client.reconnect()

# Configurar o cliente MQTT
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 120)
client.on_publish = on_publish
client.on_disconnect = on_disconnect  # Configurar a função de callback de desconexão

# Inicializa a webcam
cap = cv2.VideoCapture(0)  # 0 é o índice da webcam padrão

fgbg = cv2.createBackgroundSubtractorMOG2()

posL = 150
offset = 20
xy1 = (0, posL) 
detects = []


total = 0
entrada = 0
saida = 0

while True:
    try:
        ret, frame = cap.read()

        xy2 = (frame.shape[1], posL)  # Extremidade direita da tela (largura do quadro)
        
        cv2.line(frame, xy1, xy2, (255, 0, 0), 3)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
                    
                    # Desenhar um retângulo ao redor da pessoa detectada
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
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
                            entrada += 1
                            total += 1
                            # Publicar a contagem de pessoas no tópico MQTT
                            info_entrada = f"Entrada: {entrada}"
                            client.publish(MQTT_TOPIC1, info_entrada)
                            continue

                        if detect[c - 1][1] > posL and detect[c][1] < posL:
                            detect.clear()
                            saida += 1
                            total += 1
                            info_saida = f"Saída: {saida}"
                            client.publish(MQTT_TOPIC2, info_saida)
                            continue

        cv2.imshow("Visualização da Webcam", frame)

        # Se pressionar a tecla 'q', sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except (ConnectionError, OSError):
         print("Erro ao publicar. Reconectando...")

# Liberar a webcam e fechar a janela de visualização
cap.release()
cv2.destroyAllWindows()
