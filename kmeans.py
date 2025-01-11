import cv2
import numpy as np

def kmeans(nClusters, nRodadas, ImgOriginal, ImgDestino, numero):
    # Preparar os dados para o k-means
    samples = img.reshape((-1, 3))
    samples = np.float32(samples)

    # Aplicar k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001)
    _, labels, centers = cv2.kmeans(samples, nClusters, None, criteria, nRodadas, cv2.KMEANS_RANDOM_CENTERS)

    # Criar a imagem rotulada
    centers = np.uint8(centers)
    rotulada = centers[labels.flatten()]
    rotulada = rotulada.reshape(img.shape)

    # Redimensionar a imagem de saída
    output_size = (800, 600)  # Definir o tamanho desejado (largura, altura)
    resized_image = cv2.resize(rotulada, output_size, interpolation=cv2.INTER_LINEAR)

    # Redimensionar a imagem de entrada
    input_size = (800, 600)  # Definir o tamanho desejado (largura, altura)
    imgOrig_resize = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)

    # Exibir e salvar a imagem resultante
    cv2.imshow("ImgOriginal", imgOrig_resize)
    cv2.imshow(f"kmeans+{numero}", resized_image)
    cv2.imwrite(output_image_path, rotulada)
    cv2.waitKey(0)

nClusters = 8
nRodadas = 1

input_image_path = "imagens/flor.jpg"

# Carregar imagem
img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
if img is None:
    print("Erro ao carregar a imagem.")
    exit()

for i in range(10):
    output_image_path = f"imagens/kmenas{i}.jpg"
    kmeans(nClusters, nRodadas, img, output_image_path, i)

cv2.destroyAllWindows()

