import cv2
from Rastreador import *

seguimiento = Rastreador()
cap = cv2.VideoCapture("C:\\Users\\caco2\\Desktop\\Sin titulo2.mp4")
deteccion = cv2.createBackgroundSubtractorMOG2(history=10000, varThreshold=12)

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
            # Ensure x, y, ancho, and alto are integers and within bounds
            x, y, ancho, alto = int(x), int(y), int(ancho), int(alto)
            if x >= 0 and y >= 0 and x + ancho <= zona.shape[1] and y + alto <= zona.shape[0]:
                cv2.rectangle(zona, (x, y), (x + ancho, y + alto), (255, 255, 0), 3)
                detecciones.append([x, y, ancho, alto])

    info_id = seguimiento.rastreo(detecciones)
    for info in info_id:
        x, y, ancho, alto, identificador = info
        x, y, ancho, alto = int(x), int(y), int(ancho), int(alto)
        if x >= 0 and y >= 0 and x + ancho <= zona.shape[1] and y + alto <= zona.shape[0]:
            cv2.putText(zona, str(identificador), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
            cv2.rectangle(zona, (x, y), (x + ancho, y + alto), (255, 255, 0), 3)

    cv2.imshow("zona importante", zona)
    cv2.imshow("carretera", frame)
    cv2.imshow("mascara", mask)

    if cv2.waitKey(30) & 0xFF == 27: # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
