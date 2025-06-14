# Face Swapper App - Enhanced Version with Design, CPU/GPU Compatibility, and Stable Execution

import os
import sys
import urllib.request
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import traceback
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper

MODEL_NAME = "inswapper_128.onnx"
MODEL_URL = f"https://huggingface.co/henryruhs/insightface-models/resolve/main/{MODEL_NAME}"

# Download and ensure model exists

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def ensure_model_exists():
    model_path = resource_path(MODEL_NAME)
    if not os.path.exists(model_path):
        try:
            print("📥 Downloading model...")
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print("✅ Model downloaded.")
        except Exception as e:
            print("❌ Model download failed:", e)
            sys.exit(1)
    return model_path

class FaceSwapperApp:
    def __init__(self, root, face_app, swapper):
        self.root = root
        self.face_app = face_app
        self.swapper = swapper

        self.root.title("AI Face Swapper")
        self.root.geometry("900x700")
        self.root.configure(bg="#f7f7f7")

        self.source_img_path = ""
        self.target_img_path = ""

        tk.Label(root, text="AI Face Swapper", font=("Arial", 24, "bold"), bg="#f7f7f7").pack(pady=20)

        tk.Button(root, text="Load Source Face", font=("Arial", 14), command=self.load_source, width=30).pack(pady=5)
        tk.Button(root, text="Load Target Image", font=("Arial", 14), command=self.load_target, width=30).pack(pady=5)
        tk.Button(root, text="Swap Faces", font=("Arial", 14), command=self.swap_faces, bg="#4CAF50", fg="white", width=30).pack(pady=15)

        self.preview_label = tk.Label(root, text="Result preview will appear here", bg="#f7f7f7")
        self.preview_label.pack(pady=20)

    def load_source(self):
        self.source_img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.source_img_path:
            messagebox.showinfo("Source Loaded", f"Loaded: {self.source_img_path}")

    def load_target(self):
        self.target_img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.target_img_path:
            messagebox.showinfo("Target Loaded", f"Loaded: {self.target_img_path}")

    def swap_faces(self):
        try:
            if not self.source_img_path or not self.target_img_path:
                messagebox.showerror("Missing Images", "Please load both source and target images.")
                return

            src_img = cv2.imread(self.source_img_path)
            tgt_img = cv2.imread(self.target_img_path)

            src_faces = self.face_app.get(src_img)
            tgt_faces = self.face_app.get(tgt_img)

            if not src_faces or not tgt_faces:
                messagebox.showerror("No Faces Detected", "Make sure both images contain clear faces.")
                return

            result = self.swapper.get(tgt_img, tgt_faces[0], src_faces[0], paste_back=True)

            output_path = "swapped_result.jpg"
            cv2.imwrite(output_path, result)

            img = Image.open(output_path)
            img.thumbnail((400, 400))
            preview_img = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=preview_img)
            self.preview_label.image = preview_img

            messagebox.showinfo("Success", f"Face swapped! Saved as: {output_path}")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    try:
        model_path = ensure_model_exists()

        try:
            ctx_id = 0  # GPU
            face_app = FaceAnalysis(name="buffalo_l")
            face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        except:
            ctx_id = -1  # CPU fallback
            face_app = FaceAnalysis(name="buffalo_l")
            face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        swapper = INSwapper(model_path, ctx_id=ctx_id)

        root = tk.Tk()
        app = FaceSwapperApp(root, face_app, swapper)
        root.mainloop()

    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)
