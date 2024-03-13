import cv2
import numpy as np
from Rastreador import Rastreador

seguimiento = Rastreador()
cap = cv2.VideoCapture("C:\\Users\\caco2\\Desktop\\Sin titulo2.mp4")

# Crear el objeto Background Subtractor MOG2
deteccion = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=20,detectShadows=False)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (500, 500))

    altura_roi = 180
    y_inicio = (frame.shape[0] - altura_roi) // 2
    y_fin = y_inicio + altura_roi

    ancho_roi = 400
    x_inicio = (frame.shape[1] - ancho_roi) // 2
    x_fin = x_inicio + ancho_roi

    y_inicio = max(0, y_inicio)
    y_fin = min(frame.shape[0], y_fin)
    x_inicio = max(0, x_inicio)
    x_fin = min(frame.shape[1], x_fin)

    zona = frame[y_inicio:y_fin, x_inicio:x_fin]

    mask = deteccion.apply(zona)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    # Aplicar operaciones morfológicas para reducir el ruido
    kernel = np.ones((5, 5), np.uint8)

    mask=cv2.GaussianBlur(mask, (5, 5), 3)
    mask= cv2.erode(mask, kernel, iterations=1)
    mask= cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=1)
    _, thresholded = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    contorno, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detecciones = []
    for conto in contorno:
        area = cv2.contourArea(conto)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(conto)
            cv2.rectangle(frame, (x + x_inicio, y + y_inicio), (x + x_inicio + w, y + y_inicio + h), (255, 255, 255), 2)
            detecciones.append([x + x_inicio, y + y_inicio, w, h])

    info_id = seguimiento.rastreo(detecciones)
    for info in info_id:
        x, y, w, h, identificador = info
        cv2.putText(frame, str(identificador), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)

    print(info_id)
    cv2.imshow("zona importante", zona)
    cv2.imshow("carretera", frame)
    cv2.imshow("mascara", mask)
    cv2.imshow("umbralizado", thresholded)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
