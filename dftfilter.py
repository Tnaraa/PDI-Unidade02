#PDI - Unidade 02
#Capítulo 17.
#Usando dftfilter.cpp

#Chamar as bibliotecas
import cv2
import numpy as np

#Função swapQuadrants
def swapQuadrants(image):
    #se a imagem tiver tamanho impar, recorta a regiao para o maior
    #tamanho par possivel (-2 = 1111...1110)
    image = image[0:image.shape[0] & -2, 0:image.shape[1] & -2]

    center_x = int(image.shape[1] // 2)
    center_y = int(image.shape[0] // 2)

    #rearranja os quadrantes da transformada de Fourier de forma que
    #a origem fique no centro da imagem
    #A B   ->  D C
    #C D       B A
    A = image[0:center_y, 0:center_x]
    B = image[0:center_y, center_x:center_x*2]
    C = image[center_y:center_y*2, 0:center_x]
    D = image[center_y:center_y*2, center_x:center_x*2]

    # swap quadrants (Top-Left with Bottom-Right)
    tmp = A.copy()
    A[:] = D
    D[:] = tmp

    # swap quadrant (Top-Right with Bottom-Left)
    tmp = C.copy()
    C[:] = B
    B[:] = tmp

def makeFilter(image):
    filter2D = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    centerX, centerY = image.shape[1] // 2, image.shape[0] // 2
    radius = 20

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (i - centerY) ** 2 + (j - centerX) ** 2 <= radius ** 2:
                filter2D[i, j] = 1
            else:
                filter2D[i, j] = 0

    # Criando o filtro com as partes reais e imaginárias
    filter_real = filter2D
    filter_imag = np.zeros_like(filter2D)
    filter = cv2.merge([filter_real, filter_imag])

    return filter

#Int main:
arquivo = "imagens/senoide256x256pixels.png"

#arquivo = "imagens/biel.png"
image = cv2.imread(arquivo, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Erro abrindo imagem {arquivo}")
    exit()
#expande a imagem de entrada para o melhor tamanho no
#qual a DFT pode ser executada, preenchendo com zeros a
#lateral inferior direita.
dft_M = cv2.getOptimalDFTSize(image.shape[0])
dft_N = cv2.getOptimalDFTSize(image.shape[1])

padded = cv2.copyMakeBorder(image, 0, dft_M - image.shape[0], 0, dft_N - image.shape[1], cv2.BORDER_CONSTANT, value=0)

# prepara a matriz complexa para ser preenchida
planos = []

# prepara a matriz complexa para ser preenchida
# primeiro a parte real, contendo a imagem de entrada
planos.append(np.array(padded, dtype=np.float32))
# depois a parte imaginaria com valores nulos
planos.append(np.zeros(padded.shape, dtype=np.float32))
#planos = [np.float32(padded), np.zeros(padded.shape, dtype=np.float32)]

# combina os planos em uma unica estrutura de dados complexa
complexImage = cv2.merge(planos)

# calcula a DFT
complexImage = cv2.dft(complexImage, flags=cv2.DFT_COMPLEX_OUTPUT)
swapQuadrants(complexImage)

filter = makeFilter(complexImage)
complexImage = cv2.mulSpectrums(complexImage, filter, 0)

#Calcular dft inversa
swapQuadrants(complexImage)
complexImage = cv2.idft(complexImage)

# planos[0] : Re(DFT(image)
# planos[1] : Im(DFT(image)
planos = cv2.split(complexImage)

# Recorta a imagem filtrada para o tamanho original
result = planos[0][0:image.shape[0], 0:image.shape[1]]

# normaliza a parte real para exibicao
result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)

cv2.imshow("image", result)
cv2.imwrite("result/dft-filter.png", result * 255)

cv2.waitKey()
