import cv2
from Rastreador import Rastreador
import numpy as np

seguimiento = Rastreador()
cap = cv2.VideoCapture("videovehiculos.mp4")
deteccion = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=12)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (500, 500))
    zona = frame[230:420, 100:400]


    mask = deteccion.apply(zona)
    mask = cv2.GaussianBlur(mask, (5, 5), 1)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)


    contorno, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detecciones = []

    for conto in contorno:
        area = cv2.contourArea(conto)
        if area > 800:
            x, y, w, h = cv2.boundingRect(conto)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            detecciones.append([x, y, w, h])

    info_id = seguimiento.rastreo(detecciones)
    for info in info_id:
        x, y, ancho, alto, identificador = info
        cv2.putText(zona, str(identificador), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    )
        cv2.rectangle(zona, (x, y), (x + ancho, y + alto), (255, 255, 255), 3)

    print(info_id)
    cv2.imshow("zona importante", zona)
    cv2.imshow("carretera", frame)
    cv2.imshow("mascara", mask)

    if cv2.waitKey(30) & 0xFF == 27:  # Presiona 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()
