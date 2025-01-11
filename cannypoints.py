#PDI - Processamento Digital de Imagens
#Capítulo 19. Detecção de bordas com o algoritmo de Canny
import cv2
import random
import time

JITTER = 3
STEP = 3

def cannyfilter(image, val):
    imgCanny = cv2.Canny(image, val, 3 * val)
    return imgCanny

def pontilhismo(image, imagecolor, raio):
    imgponto = imagecolor.copy()
    height, width = image.shape

    xrange = list(range(height // STEP))
    yrange = list(range(width // STEP))

    for i in range(len(xrange)):
        xrange[i] = xrange[i] * STEP + STEP // 2

    for i in range(len(yrange)):
        yrange[i] = yrange[i] * STEP + STEP // 2

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
            if x >= 0 and x < image.shape[0] and y >= 0 and y < image.shape[1]:
                pixel_value = imagecolor[x, y]
                blue, green, red = pixel_value
                color = (int(blue), int(green), int(red))
                # Desenhar um círculo no ponto correspondente
                cv2.circle(imgponto, (y, x), raio, color , -1, cv2.LINE_AA)
   # cv2.imshow("teste", imgponto)
    return imgponto

def desenha(imgPonti, imagecolor, imgCanny, p):
    imgCannyPoint = imgPonti.copy()
    for i in range(imgPonti.shape[0]):
        for j in range(imgPonti.shape[1]):
            if (imgCanny[i][j] == 255):
                #gray = int(image[i][j])
                pixel_value = imagecolor[i,j]
                blue, green, red = pixel_value
                color = (int(blue), int(green), int(red))
                cv2.circle(imgCannyPoint, (j, i), p, color, p)
    return imgCannyPoint

#Int main:
arquivo = "imagens/sertao2.jpg"
imagecolor = cv2.imread(arquivo, cv2.IMREAD_COLOR)
image = cv2.imread(arquivo, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Erro abrindo imagem {arquivo}")
    exit()

cv2.imshow("Original", imagecolor)
raio = 10
for qnt in range(1, 100, 10):
    imgCanny = cannyfilter(image, qnt*2)

    cv2.imshow(f"Canny{qnt}", imgCanny)
    cv2.imwrite(f"imagens/filtercanny{qnt}.png", imgCanny)

    imgp = pontilhismo(image, imagecolor, raio+1)
    imgCannyPoint = desenha(imgp, imagecolor, imgCanny, raio)

    cv2.imshow(f"cannypoint{qnt}", imgCannyPoint)
    cv2.imwrite(f"imagens/canny&point{qnt}.png", imgCannyPoint)

    raio = raio - 1
    if raio <= 0:
        raio = 1

cv2.imshow("cannypoint", imgCannyPoint)
cv2.imwrite("imagens/bordaspontilhismo.png", imgCanny)
cv2.imwrite("imagens/pontilhismo.png", imgp)
cv2.imwrite("imagens/PontilhismoeCanny.png", imgCannyPoint)
cv2.waitKey(0)
cv2.destroyAllWindows()
