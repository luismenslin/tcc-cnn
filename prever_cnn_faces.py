import sys
import json
import os
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2

MODEL_PATH = "modelo_faces.keras"
CLASSES_PATH = "classes.json"
IMG_SIZE = 224

detector = MTCNN()
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)


# ============================================================
# Função para prever rosto em imagem PIL
# ============================================================
def predict_face_pil(img_pil):
    img = np.array(img_pil.convert("RGB"))
    dets = detector.detect_faces(img)
    if not dets:
        return None, None, None

    # maior rosto
    det = max(dets, key=lambda d: d['box'][2] * d['box'][3])
    x, y, w, h = det['box']
    x, y = max(0, x), max(0, y)

    face = img[y:y + h, x:x + w]
    if face.size == 0:
        return None, None, None

    face_pil = Image.fromarray(face).resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(face_pil, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))

    return CLASS_NAMES[idx], float(probs[idx]), det


# ============================================================
# Predição em imagem estática
# ============================================================
def predict_image(path, threshold=0.75):
    # Lê a imagem com OpenCV
    frame = cv2.imread(path)
    if frame is None:
        print("❌ Não foi possível abrir a imagem:", path)
        return

    # Converte BGR -> RGB para o MTCNN
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector.detect_faces(rgb)

    if not dets:
        print("❌ Nenhum rosto detectado.")
        cv2.imshow("Reconhecimento Facial (Imagem / CNN)", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    for d in dets:
        x, y, w, h = d['box']
        x, y = max(0, x), max(0, y)
        face = rgb[y:y+h, x:x+w]

        if face.size == 0:
            continue

        # Prepara o recorte pro modelo CNN
        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        arr = preprocess_input(face_resized.astype(np.float32))
        arr = np.expand_dims(arr, 0)

        probs = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        name = CLASS_NAMES[idx]
        conf = float(probs[idx])

        label = f"{name} ({conf:.2f})" if conf >= threshold else "Desconhecido"
        color = (0, 255, 0) if conf >= threshold else (0, 0, 255)

        # Desenha retângulo e texto na imagem original
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Mostra a imagem em uma janela
    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
    cv2.imshow("Reconhecimento Facial (Imagem / CNN)", frame)
    print(f"✅ Predição exibida na janela. Pressione qualquer tecla para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Predição em vídeo (arquivo .mp4, .avi, etc.)
# ============================================================
def predict_video(video_path, threshold=0.75):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Não foi possível abrir o vídeo:", video_path)
        return

    print("🎥 Processando vídeo... (ESC para sair)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector.detect_faces(rgb)

        for d in dets:
            x, y, w, h = d['box']
            x, y = max(0, x), max(0, y)
            face = rgb[y:y + h, x:x + w]

            if face.size == 0:
                continue

            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            arr = preprocess_input(face.astype(np.float32))
            arr = np.expand_dims(arr, 0)

            probs = model.predict(arr, verbose=0)[0]
            idx = int(np.argmax(probs))
            name = CLASS_NAMES[idx]
            conf = float(probs[idx])

            label = f"{name} ({conf:.2f})" if conf >= threshold else "Desconhecido"
            color = (0, 255, 0) if conf >= threshold else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Reconhecimento Facial (Vídeo / CNN)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Predição Webcam
# ============================================================
def predict_webcam(cam_index=0, threshold=0.75):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Não foi possível abrir a webcam.")
        return

    print("📷 Webcam ativada (ESC para sair)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector.detect_faces(rgb)

        for d in dets:
            x, y, w, h = d['box']
            x, y = max(0, x), max(0, y)
            face = rgb[y:y + h, x:x + w]

            if face.size == 0:
                continue

            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            arr = preprocess_input(face.astype(np.float32))
            arr = np.expand_dims(arr, 0)

            probs = model.predict(arr, verbose=0)[0]
            idx = int(np.argmax(probs))
            name = CLASS_NAMES[idx]
            conf = float(probs[idx])

            label = f"{name} ({conf:.2f})" if conf >= threshold else "Desconhecido"
            color = (0, 255, 0) if conf >= threshold else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Reconhecimento Facial (Webcam / CNN)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Uso:")
        print("  python prever_cnn_faces.py caminho.jpg")
        print("  python prever_cnn_faces.py --webcam")
        print("  python prever_cnn_faces.py --video caminho_do_video.mp4")
        sys.exit(0)

    if sys.argv[1] == "--webcam":
        predict_webcam()

    elif sys.argv[1] == "--video":
        if len(sys.argv) < 3:
            print("Erro: informe o caminho do vídeo")
            sys.exit(1)
        predict_video(sys.argv[2])

    else:
        predict_image(sys.argv[1])
