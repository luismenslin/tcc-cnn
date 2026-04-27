import os
import json
import cv2
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


MODEL_PATH = "modelo_faces.keras"
CLASSES_PATH = "classes.json"
IMG_SIZE = 224
THRESHOLD = 0.75

detector = MTCNN()
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)


def reconhecer_imagem(path):
    frame = cv2.imread(path)

    if frame is None:
        return None, "Erro ao abrir imagem"

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector.detect_faces(rgb)

    if not dets:
        return frame, "Nenhum rosto detectado"

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

    return frame, "Reconhecimento concluído"


def redimensionar_para_tela(frame, max_width=800, max_height=550):
    h, w = frame.shape[:2]

    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

    return frame


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Teste de Reconhecimento Facial - CNN")
        self.root.geometry("950x750")

        self.imagens = []
        self.indice_atual = 0
        self.tk_image = None

        titulo = tk.Label(
            root,
            text="Reconhecimento Facial com CNN",
            font=("Arial", 18, "bold")
        )
        titulo.pack(pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.btn_selecionar = tk.Button(
            btn_frame,
            text="Selecionar imagem(ns)",
            command=self.selecionar_imagens,
            font=("Arial", 12),
            width=22
        )
        self.btn_selecionar.grid(row=0, column=0, padx=5)

        self.btn_anterior = tk.Button(
            btn_frame,
            text="Anterior",
            command=self.anterior,
            font=("Arial", 12),
            width=12
        )
        self.btn_anterior.grid(row=0, column=1, padx=5)

        self.btn_proxima = tk.Button(
            btn_frame,
            text="Próxima",
            command=self.proxima,
            font=("Arial", 12),
            width=12
        )
        self.btn_proxima.grid(row=0, column=2, padx=5)

        self.lbl_status = tk.Label(
            root,
            text="Nenhuma imagem selecionada",
            font=("Arial", 11)
        )
        self.lbl_status.pack(pady=5)

        self.lbl_imagem = tk.Label(root)
        self.lbl_imagem.pack(pady=10)

    def selecionar_imagens(self):
        paths = filedialog.askopenfilenames(
            title="Selecione uma ou mais imagens",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Todos os arquivos", "*.*")
            ]
        )

        if not paths:
            return

        self.imagens = list(paths)
        self.indice_atual = 0
        self.exibir_imagem_atual()

    def exibir_imagem_atual(self):
        if not self.imagens:
            return

        path = self.imagens[self.indice_atual]
        frame, status = reconhecer_imagem(path)

        if frame is None:
            messagebox.showerror("Erro", status)
            return

        frame = redimensionar_para_tela(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(frame_rgb)
        self.tk_image = ImageTk.PhotoImage(img_pil)

        self.lbl_imagem.config(image=self.tk_image)

        nome_arquivo = os.path.basename(path)
        self.lbl_status.config(
            text=f"{self.indice_atual + 1}/{len(self.imagens)} - {nome_arquivo} - {status}"
        )

    def proxima(self):
        if not self.imagens:
            return

        self.indice_atual = (self.indice_atual + 1) % len(self.imagens)
        self.exibir_imagem_atual()

    def anterior(self):
        if not self.imagens:
            return

        self.indice_atual = (self.indice_atual - 1) % len(self.imagens)
        self.exibir_imagem_atual()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()