#PDI - Unidade 02
#Capítulo 17. Filtragem no Domínio da Frequência
#Usando dftfilter.cpp, criar um filtro homomorfico

#Chamar as bibliotecas
import cv2
import numpy as np

imageaux = None
image = None
gamaL = 5
gamaL_max = 100

gamaH_max = 100
gamaH = 5

c_slider = 5
c_slider_max = 100

D0_slider = 5
D0_slider_max = 100

TrackbarName = ['']*50

#Funções auxiliares para atualizar os valores
def on_trackbar_c(val):
    global c_slider
    c_slider = val
    on_trackbar_functions(image)

def on_trackbar_gamaH(val):
    global gamaH
    gamaH = val
    on_trackbar_functions(image)

def on_trackbar_gamaL(val):
    global gamaL
    gamaL = val
    on_trackbar_functions(image)

def on_trackbar_D0(val):
    global D0_slider
    D0_slider = val
    on_trackbar_functions(image)

#Realizar a conversão/condições dos valores dos parametros obtidos via trackbar
def on_trackbar_functions(imageaux):
    gL = gamaL/10
    gH = gamaH/10

    if c_slider > 0: c = c_slider
    else: c = 1

    if D0_slider == 0: D0 = 1
    else: D0 = D0_slider

    print(f'gamaH: {gH}')
    print(f'gamaL: {gL}')
    print(f'inclincação c: {c}')
    print(f'D0: {D0}')

    imagefiltrada = filtro_Homomorfico(imageaux, gL, gH, c, D0)

    # Normaliza e exibe a imagem
    img_normalizada = cv2.normalize(imagefiltrada, None, 0, 255, cv2.NORM_MINMAX)
    img_normalizada = np.uint8(img_normalizada)
    cv2.imshow('Homomorfico', img_normalizada)
    cv2.imshow("image", imageaux)
    cv2.imwrite('imagens/homomorficofilter.png', img_normalizada)


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

def makeFilter(image, gL, gH, c, D0):
    filter2D = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    centerX, centerY = image.shape[1] // 2, image.shape[0] // 2

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            distance = (i - centerY) ** 2 + (j - centerX) ** 2
          #  filter2D[i, j] = (gH - gL) * (1 - np.exp(-distance / (2 * c ** 2))) + gL
            filter2D[i,j] = (gH - gL) * (1 - np.exp(-c*((distance**2)/(D0**2)))) + gL

    # Criando o filtro com as partes reais e imaginárias
    filter_real = filter2D
    filter_imag = np.zeros_like(filter2D)
    filter = cv2.merge([filter_real, filter_imag])

    return filter

def filtro_Homomorfico(image, gL, gH, c, D0):
    dft_M = cv2.getOptimalDFTSize(image.shape[0])
    dft_N = cv2.getOptimalDFTSize(image.shape[1])

    padded = cv2.copyMakeBorder(image, 0, dft_M - image.shape[0], 0, dft_N - image.shape[1],
                                cv2.BORDER_CONSTANT,
                                value=0)

    # prepara a matriz complexa para ser preenchida
    planos = []
    # prepara a matriz complexa para ser preenchida
    # primeiro a parte real, contendo a imagem de entrada
    planos.append(np.array(padded, dtype=np.float32))
    # depois a parte imaginaria com valores nulos
    planos.append(np.zeros(padded.shape, dtype=np.float32))
    # planos = [np.float32(padded), np.zeros(padded.shape, dtype=np.float32)]

    # combina os planos em uma unica estrutura de dados complexa
    complexImage = cv2.merge(planos)

    # calcula a DFT
    complexImage = cv2.dft(complexImage, flags=cv2.DFT_COMPLEX_OUTPUT)
    swapQuadrants(complexImage)

    filter = makeFilter(complexImage, gL, gH, c, D0)
    complexImage = cv2.mulSpectrums(complexImage, filter, 0)

    # Calcular dft inversa
    swapQuadrants(complexImage)
    complexImage = cv2.idft(complexImage)

    # planos[0] : Re(DFT(image)
    # planos[1] : Im(DFT(image)
    planos = cv2.split(complexImage)

    # Recorta a imagem filtrada para o tamanho original
    result = planos[0][0:image.shape[0], 0:image.shape[1]]

    # normaliza a parte real para exibicao
    result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)
    return result

#Int main:
arquivo = "imagens/fotopendrive.jpg"

#arquivo = "imagens/biel.png"
image = cv2.imread(arquivo, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Erro abrindo imagem {arquivo}")
    exit()
size = (600, 500)  # Definir o tamanho desejado (largura, altura)
image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    # image = cv2.resize(imagem, size, interpolation=cv2.INTER_LINEAR)
imageaux = np.copy(image) #Para não mexer com a imagem

cv2.namedWindow("Homomorfico", 1)

TrackbarName = 'gamaH'
cv2.createTrackbar(TrackbarName, "Homomorfico",
                   gamaH,
                   gamaH_max,
                   on_trackbar_gamaH)

TrackbarName = 'gamaL'
cv2.createTrackbar(TrackbarName, "Homomorfico",
                   gamaL,
                   gamaL_max,
                   on_trackbar_gamaL)

TrackbarName = 'c x %d' % c_slider_max
cv2.createTrackbar(TrackbarName, "Homomorfico",
                   c_slider,
                   c_slider_max,
                   on_trackbar_c)

TrackbarName = 'D0 x %d' % D0_slider_max
cv2.createTrackbar(TrackbarName, "Homomorfico",
                   D0_slider,
                   D0_slider_max,
                   on_trackbar_D0)

on_trackbar_functions(imageaux)

cv2.waitKey()
