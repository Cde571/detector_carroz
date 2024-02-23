from rembg import remove
import cv2
import numpy as np

# Función para eliminar el fondo de un fotograma
def remove_background(frame):
    _, img_data = cv2.imencode(".png", frame)
    img_data_without_bg = remove(img_data.tobytes())
    img_np_without_bg = cv2.imdecode(np.frombuffer(img_data_without_bg, np.uint8), cv2.IMREAD_COLOR)
    return img_np_without_bg

# Cargar el video
cap = cv2.VideoCapture("videovehiculos.mp4")

# Ejemplo: Detector de carros usando Haarcascades
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Dentro del bucle de captura de fotogramas
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Aplicar el removedor de fondo
    frame_without_bg = remove_background(frame)

    # Convertir a escala de grises para el detector de carros
    gray = cv2.cvtColor(frame_without_bg, cv2.COLOR_BGR2GRAY)

    # Detectar carros
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

    # Dibujar rectángulos alrededor de los carros detectados
    for (x, y, w, h) in cars:
        cv2.rectangle(frame_without_bg, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Aquí puedes agregar lógica para identificar el tipo de carro y buscar información en internet
        # ...

    cv2.imshow("Fotograma sin Fondo con Carros", frame_without_bg)

    if cv2.waitKey(30) & 0xFF == 27:  # Presiona 'Esc' para salir
        break

# Liberar la captura y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
