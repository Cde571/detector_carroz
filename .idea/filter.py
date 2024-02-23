import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread("sin_fondo.png")
imagen = cv2.resize(imagen, (500, 500))

# Mostrar la imagen en una ventana llamada "Imagen"
cv2.imshow("Imagen", imagen)

# Esperar a que se presione una tecla y luego cerrar la ventana
cv2.waitKey(0)
cv2.destroyAllWindows()

# Función que muestra información sobre la imagen
def howis(imagen):
    print('size=', imagen.shape)
    print('dtype=', imagen.dtype)
    print('min=', np.min(imagen))
    print('max=', np.max(imagen))

# Obtener el canal azul de la imagen
X = imagen[:, :, 0]
howis(X)
cv2.imshow("Canal Azul", X)

# Función para segmentar la imagen
def segmenta(X, t):
    (N, M) = X.shape
    Y = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            if X[i, j] < t:
                Y[i, j] = 0
            else:
                Y[i, j] = 255
    return Y

# Segmentar el canal azul con un umbral de 100
Y = segmenta(X, 100)
howis(Y)
cv2.imshow("Imagen Segmentada", Y)

# Esperar a que se presione una tecla y luego cerrar la ventana
cv2.waitKey(0)
cv2.destroyAllWindows()
