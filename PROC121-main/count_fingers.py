# Importe as bibliotecas necessárias
import cv2
import mediapipe as mp

# Inicialize a captura de vídeo a partir da câmera (0 representa a câmera padrão)
cap = cv2.VideoCapture(0)

# Inicialize o módulo Hands do MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure o modelo Hands com valores mínimos de confiança para detecção e rastreamento
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

# Defina os índices das pontas dos dedos
tipIds = [0, 4, 8, 12, 16, 20]

def countFingers(image, hand_landmarks, handNo=0):
    # defina a função aqui
    landmarks = hand_landmarks[handNo].landmark

    #lista para contar os dedos
    fingers = []

    for lm_index in tipIds:
        fingerId = int((lm_index-1)/4)
        # obtenha as coordenadas Y da ponta e da base do dedo
        finger_tip_y = landmarks[lm_index].y
        finger_bottom_y = landmarks[lm_index - 2].y
        #Verifique se o dedo está aberto ou fechado
        if finger_tip_y < finger_bottom_y:
            fingers.append(1)
        else:
            fingers.append(0)
        #conte o total de dedos abertos
        totalFingers = fingers.count(1)

        #exiba o texto com o número de dedos abertos
        text = f'Dedos: {totalFingers}'
        cv2.putText(image, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

def drawHandLandmarks(image, hand_landmarks):
    # Defina uma função para desenhar os pontos de referência da mão aqui
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp.hands.HAND_CONNECTIONS)

# Loop principal
while True:
    # Captura um quadro da câmera
    success, image = cap.read()

    # Inverte a imagem horizontalmente para que seja mais intuitivo
    image = cv2.flip(image, 1)
    
    # Detecte os pontos de referência das mãos 
    results = hands.process(image)

    # Obtenha a posição dos pontos de referência da mão
    hand_landmarks = results.multi_hand_landmarks

    # Desenhe os pontos de referência na imagem
    drawHandLandmarks(image, hand_landmarks)

    # Conte os dedos e exiba o resultado
    countFingers(image, hand_landmarks)

    # Exiba a imagem com as informações
    cv2.imshow("Controlador de Mídia", image)

    # Saia do programa ao pressionar a tecla 'q'
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Libere a captura de vídeo e feche as janelas OpenCV
cv2.destroyAllWindows()
cap.release()
