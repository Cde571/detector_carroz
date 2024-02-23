import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread("C:\\Users\\caco2\\Desktop\\output\\sin_fondo.png")
imagen = cv2.resize(imagen, (500, 500))

# Convertir la imagen a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplicar umbral para resaltar los píxeles blancos
_, umbral = cv2.threshold(gris, 200, 255, cv2.THRESH_BINARY)

# Encontrar contornos en la imagen umbralizada
contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar contornos en la imagen original
imagen_contornos = imagen.copy()
cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)

# Mostrar la imagen original, la imagen umbralizada y la imagen con contornos
cv2.imshow("Imagen Original", imagen)
cv2.imshow("Imagen Umbralizada", umbral)
cv2.imshow("Imagen con Contornos", imagen_contornos)
cv2.waitKey(0)
cv2.destroyAllWindows()
