import os, json, glob, math, random
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===== Configurações =====
DATASET_DIR = "dataset"
IMG_SIZE = 224
VAL_SPLIT = 0.2
BATCH_SIZE = 16
EPOCHS = 25
OUTPUT_MODEL = "modelo_faces.keras"
OUTPUT_CLASSES = "classes.json"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ===== Detecção e recorte de rostos com MTCNN (CNN) =====
detector = MTCNN()

def detect_and_crop_face(img_pil, margin=0.2):
    """Detecta o rosto principal e retorna recorte alinhado (aprox) como PIL.Image."""
    img = np.array(img_pil.convert("RGB"))
    detections = detector.detect_faces(img)

    if not detections:
        return None

    # pega a maior detecção (mais provável ser o rosto principal)
    det = max(detections, key=lambda d: d['box'][2] * d['box'][3])
    x, y, w, h = det['box']
    x, y = max(0, x), max(0, y)

    # margem ao redor do rosto
    mx = int(w * margin)
    my = int(h * margin)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(img.shape[1], x + w + mx), min(img.shape[0], y + h + my)

    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return None

    # redimensiona
    face_pil = Image.fromarray(face).resize((IMG_SIZE, IMG_SIZE))
    return face_pil
def load_dataset(dataset_dir):
    X, y, paths_ok, paths_fail = [], [], [], []
    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    if not class_names:
        raise RuntimeError("Nenhuma classe encontrada em 'dataset/'. Crie subpastas por pessoa.")

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print("[DEBUG] Pastas/classes detectadas:", class_names)

    # extensões permitidas (case-insensitive)
    exts_validas = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for cls in class_names:
        folder = os.path.join(dataset_dir, cls)

        # lista todos os arquivos da pasta
        arquivos = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        ]

        # filtra só imagens pelas extensões (lower)
        images = [
            p for p in arquivos
            if os.path.splitext(p)[1].lower() in exts_validas
        ]

        if not images:
            print(f"[AVISO] Sem imagens (ou extensões incompatíveis) para: {cls}")
            continue

        for path in images:
            try:
                img = Image.open(path)
                face = detect_and_crop_face(img)
                if face is None:
                    paths_fail.append(path)
                    continue

                arr = np.array(face, dtype=np.float32)
                arr = preprocess_input(arr)  # MobileNetV2
                X.append(arr)
                y.append(class_to_idx[cls])
                paths_ok.append(path)
            except Exception as e:
                print(f"[ERRO] {path}: {e}")
                paths_fail.append(path)

    if len(X) == 0:
        raise RuntimeError(
            "Nenhuma imagem válida foi carregada. "
            "Verifique se há arquivos de imagem (jpg, jpeg, png, bmp, webp) "
            "diretamente dentro das pastas de cada pessoa em 'dataset/'."
        )

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int32)
    print(f"[INFO] Imagens OK: {len(paths_ok)} | Falhas (detecção/erro): {len(paths_fail)}")

    return X, y, class_names


def build_model(num_classes):
    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False  # congela no início (pode destravar para fine-tune depois)

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def make_tfdata(X, y, training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        # pequenas augments (não exagerar para não distorcer rostos)
        def aug(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.9, 1.1)
            return img, label
        ds = ds.shuffle(len(X), seed=RANDOM_SEED).map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def main():
    X, y, class_names = load_dataset(DATASET_DIR)

    # split estratificado
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, stratify=y, random_state=RANDOM_SEED
    )

    train_ds = make_tfdata(X_train, y_train, training=True)
    val_ds   = make_tfdata(X_val, y_val, training=False)

    model = build_model(num_classes=len(class_names))

    cbs = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(OUTPUT_MODEL, save_best_only=True, monitor="val_accuracy")
    ]

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=1
    )

    # opcional: fine-tune leve (descongela o topo da base)
    base = model.layers[1]
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy")
        ],
        verbose=1
    )

    model.save(OUTPUT_MODEL)
    with open(OUTPUT_CLASSES, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # métrica final
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"[RESULTADO] Val Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()
