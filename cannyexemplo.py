import cv2
import numpy as np

# Variáveis globais
top_slider = 10
top_slider_max = 200
image = None  # Declarando image como global


# Função callback chamada a cada vez que o slider é movido
def on_trackbar_canny(val):
    global image  # Definindo a variável global 'image'
    # Atualiza o limiar superior como 3 vezes o valor do limiar inferior
    upper_threshold = 3 * val
    # Aplica o algoritmo Canny
    border = cv2.Canny(image, val, upper_threshold)

    # Verifica se a imagem com as bordas foi gerada corretamente
    if border is not None and border.size > 0:
        # Exibe a imagem de bordas
        cv2.imshow("Canny", border)
    else:
        print("Erro ao aplicar Canny. Verifique a imagem.")


def main():
    global image  # Definindo a variável global 'image'

    # Carregar a imagem em escala de cinza
    image = cv2.imread("imagens/sertao2.jpg", cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Could not open or find the image")
        return

    # Cria uma janela
    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)

    # Cria um trackbar para controlar o limiar inferior
    cv2.createTrackbar("Threshold Inferior", "Canny", top_slider, top_slider_max, on_trackbar_canny)

    # Chama a função uma vez para inicializar a visualização
    on_trackbar_canny(top_slider)

    # Aguardar a tecla pressionada
    cv2.waitKey(0)

    # Salvar a imagem de bordas
    border = cv2.Canny(image, top_slider, 3 * top_slider)
    if border is not None and border.size > 0:
        cv2.imwrite("cannyborders.png", border)
    else:
        print("Erro ao salvar a imagem com bordas.")

    # Fechar todas as janelas abertas
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




