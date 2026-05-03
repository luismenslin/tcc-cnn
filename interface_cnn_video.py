import cv2
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json

import tkinter as tk
from tkinter import filedialog

# ================= CONFIG =================
MODEL_PATH = "modelo_faces.keras"
CLASSES_PATH = "classes.json"
IMG_SIZE = 224
THRESHOLD = 0.75

# ================= LOAD =================
detector = MTCNN()
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)


# ================= RECONHECIMENTO =================
def reconhecer_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector.detect_faces(rgb)

    for d in dets:
        x, y, w, h = d["box"]
        x, y = max(0, x), max(0, y)

        face = rgb[y:y+h, x:x+w]

        if face.size == 0:
            continue

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        arr = preprocess_input(face.astype(np.float32))
        arr = np.expand_dims(arr, axis=0)

        probs = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        nome = CLASS_NAMES[idx]
        conf = float(probs[idx])

        if conf >= THRESHOLD:
            label = f"{nome} ({conf:.2f})"
            color = (0, 255, 0)
        else:
            label = f"Desconhecido ({conf:.2f})"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    return frame


# ================= APP =================
class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecimento Facial - Vídeo (CNN)")
        self.root.geometry("400x150")

        self.cap = None
        self.rodando = False

        tk.Label(
            root,
            text="Reconhecimento Facial em Vídeo (CNN)",
            font=("Arial", 12, "bold")
        ).pack(pady=10)

        tk.Button(
            root,
            text="Selecionar Vídeo",
            command=self.selecionar_video,
            width=25
        ).pack(pady=5)

        tk.Button(
            root,
            text="Parar",
            command=self.parar,
            width=25
        ).pack(pady=5)

    def selecionar_video(self):
        path = filedialog.askopenfilename(
            title="Selecione um vídeo",
            filetypes=[("Vídeos", "*.mp4 *.avi *.mov *.mkv")]
        )

        if not path:
            return

        self.cap = cv2.VideoCapture(path)
        self.rodando = True
        self.processar()

    def processar(self):
        if not self.rodando or self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.parar()
            return

        frame = reconhecer_frame(frame)

        cv2.imshow("Reconhecimento - CNN", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            self.parar()
            return

        self.root.after(10, self.processar)

    def parar(self):
        self.rodando = False

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()


# ================= MAIN =================
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()