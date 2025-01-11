import cv2
import numpy as np
import random
import time

# Constantes
STEP = 5
JITTER = 3
RAIO = 3

def main():
    # Carregar a imagem
    image = cv2.imread("imagens/natura.jpg", cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("A imagem não foi aberta")
        return

    height, width = image.shape

    xrange = list(range(height // STEP))
    yrange = list(range(width // STEP))

    for i in range(len(xrange)):
        xrange[i] = xrange[i] * STEP + STEP // 2

    for i in range(len(yrange)):
        yrange[i] = yrange[i] * STEP + STEP // 2

    # Criar uma imagem de pontos em branco
    points = np.ones((height, width), dtype=np.uint8) * 255

    # Gerar uma semente aleatória para o embaralhamento
    seed = int(time.time() * 1000)  # Usando a hora atual para gerar uma semente única
    random.seed(seed)
    random.shuffle(xrange)

    for i in xrange:
        random.shuffle(yrange)
        for j in yrange:
            # Adicionar jitter aleatório
            x = i + random.randint(-JITTER, JITTER)
            y = j + random.randint(-JITTER, JITTER)

            # Garantir que x e y estão dentro dos limites da imagem
            if x >= 0 and x < height and y >= 0 and y < width:
                gray = int(image[x, y])
                # Desenhar um círculo no ponto correspondente (escala de cinza)
                cv2.circle(points, (y, x), RAIO, (gray, gray, gray), -1, cv2.LINE_AA)

    # Salvar a imagem gerada
    cv2.imshow("pontos", points)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()
