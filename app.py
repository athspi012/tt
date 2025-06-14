import os
import sys
import urllib.request
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

MODEL_NAME = "inswapper_128.onnx"
MODEL_URL = "https://huggingface.co/henryruhs/insightface-models/resolve/main/" + MODEL_NAME

def resource_path(relative_path):
    # for PyInstaller
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def ensure_model_exists():
    model_path = resource_path(MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"Downloading model from {MODEL_URL}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print("✅ Model downloaded.")
        except Exception as e:
            print("❌ Failed to download model:", e)
            sys.exit(1)
    return model_path

class FaceSwapperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Face Swapper")
        self.root.geometry("800x600")
        self.source_img_path = ""
        self.target_img_path = ""

        tk.Button(root, text="Load Source Face", command=self.load_source).pack(pady=10)
        tk.Button(root, text="Load Target Image", command=self.load_target).pack(pady=10)
        tk.Button(root, text="Swap Faces", command=self.swap_faces).pack(pady=10)
        self.preview_label = tk.Label(root, text="Result will show here")
        self.preview_label.pack(pady=20)

    def load_source(self):
        self.source_img_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if self.source_img_path:
            messagebox.showinfo("Source Image Loaded", self.source_img_path)

    def load_target(self):
        self.target_img_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if self.target_img_path:
            messagebox.showinfo("Target Image Loaded", self.target_img_path)

    def swap_faces(self):
        if not self.source_img_path or not self.target_img_path:
            messagebox.showerror("Error", "Load both source and target images first.")
            return

        src_img = cv2.imread(self.source_img_path)
        tgt_img = cv2.imread(self.target_img_path)
        src_faces = face_app.get(src_img)
        tgt_faces = face_app.get(tgt_img)

        if not src_faces or not tgt_faces:
            messagebox.showerror("Error", "No face detected in one of the images.")
            return

        result_img = swapper.get(tgt_img, tgt_faces[0], src_faces[0], paste_back=True)
        output_path = "swapped_result.jpg"
        cv2.imwrite(output_path, result_img)
        img = Image.open(output_path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=img_tk)
        self.preview_label.image = img_tk
        messagebox.showinfo("Success", "Face swapped image saved as swapped_result.jpg")

if __name__ == "__main__":
    model_path = ensure_model_exists()
    face_app = FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = get_model(model_path, download=False, download_zip=False)

    root = tk.Tk()
    app = FaceSwapperApp(root)
    root.mainloop()
