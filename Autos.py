import cv2
from Rastreador import *

seguimiento = Rastreador()
cap = cv2.VideoCapture("C:\\Users\\Cristian\\PycharmProjects\\detector-autos\\Sin titulo2.mp4")
deteccion = cv2.createBackgroundSubtractorMOG2(history=10000, varThreshold=12)

# Diccionario para almacenar la posición anterior de cada auto
posicion_anterior = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break # Break the loop if no frame is read

    frame = cv2.resize(frame, (432, 500))
    zona = frame[230:330, 100:400]

    mask = deteccion.apply(zona)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contorno, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detecciones = []

    for conto in contorno:
        area = cv2.contourArea(conto)
        if area > 800:
            x, y, ancho, alto = cv2.boundingRect(conto)
            x, y, ancho, alto = int(x), int(y), int(ancho), int(alto)
            if x >= 0 and y >= 0 and x + ancho <= zona.shape[1] and y + alto <= zona.shape[0]:
                cv2.rectangle(zona, (x, y), (x + ancho, y + alto), (255, 255, 0), 3)
                detecciones.append([x, y, ancho, alto])

    info_id = seguimiento.rastreo(detecciones)
    for info in info_id:
        x, y, ancho, alto, identificador = info
        x, y, ancho, alto = int(x), int(y), int(ancho), int(alto)
        if x >= 0 and y >= 0 and x + ancho <= zona.shape[1] and y + alto <= zona.shape[0]:
            # Calcular la velocidad si la posición anterior está disponible
            if identificador in posicion_anterior:
                posicion_anterior_x, posicion_anterior_y = posicion_anterior[identificador]
                distancia = ((x - posicion_anterior_x)**2 + (y - posicion_anterior_y)**2)**0.5
                # Asumiendo una distancia fija entre frames para calcular la velocidad
                velocidad = distancia / 0.033 # 0.033 es un ejemplo de tiempo entre frames en segundos
                cv2.putText(zona, f"Vel: {velocidad:.2f} px/s", (x, y - 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            else:
                velocidad = 0
            cv2.putText(zona, str(identificador), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
            cv2.rectangle(zona, (x, y), (x + ancho, y + alto), (255, 255, 0), 3)
            # Actualizar la posición anterior
            posicion_anterior[identificador] = (x, y)

    cv2.imshow("zona importante", zona)
    cv2.imshow("carretera", frame)
    cv2.imshow("mascara", mask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
