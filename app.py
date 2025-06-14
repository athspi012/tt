# Face Swapper App - Full Stable Version with Enhanced Design and GPU/CPU Compatibility

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
from insightface.model_zoo import get_model

MODEL_NAME = "inswapper_128.onnx"
MODEL_URL = f"https://huggingface.co/henryruhs/insightface-models/resolve/main/{MODEL_NAME}"

# Safe model downloader
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def ensure_model_exists():
    model_path = resource_path(MODEL_NAME)
    if not os.path.exists(model_path):
        try:
            print("Downloading AI model...")
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print("Model downloaded successfully.")
        except Exception as e:
            print("Failed to download model:", e)
            sys.exit(1)
    return model_path

class FaceSwapperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔁 AI Face Swapper")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")

        self.source_img_path = ""
        self.target_img_path = ""

        tk.Label(root, text="AI Face Swapper", font=("Arial", 24, "bold"), bg="#f0f0f0").pack(pady=20)

        tk.Button(root, text="📤 Load Source Face", font=("Arial", 14), command=self.load_source, width=25).pack(pady=10)
        tk.Button(root, text="🖼️ Load Target Image", font=("Arial", 14), command=self.load_target, width=25).pack(pady=10)
        tk.Button(root, text="🚀 Swap Faces", font=("Arial", 14, "bold"), command=self.swap_faces, bg="#4CAF50", fg="white", width=25).pack(pady=20)

        self.preview_label = tk.Label(root, text="Preview will appear here", bg="#f0f0f0")
        self.preview_label.pack(pady=20)

    def load_source(self):
        self.source_img_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if self.source_img_path:
            messagebox.showinfo("Loaded", f"Source image loaded: {os.path.basename(self.source_img_path)}")

    def load_target(self):
        self.target_img_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if self.target_img_path:
            messagebox.showinfo("Loaded", f"Target image loaded: {os.path.basename(self.target_img_path)}")

    def swap_faces(self):
        try:
            if not self.source_img_path or not self.target_img_path:
                messagebox.showerror("Error", "Please load both source and target images.")
                return

            src_img = cv2.imread(self.source_img_path)
            tgt_img = cv2.imread(self.target_img_path)

            src_faces = face_app.get(src_img)
            tgt_faces = face_app.get(tgt_img)

            if not src_faces or not tgt_faces:
                messagebox.showerror("Error", "No faces detected in one or both images.")
                return

            result_img = swapper.get(tgt_img, tgt_faces[0], src_faces[0], paste_back=True)
            output_path = "swapped_output.jpg"
            cv2.imwrite(output_path, result_img)

            img = Image.open(output_path)
            img.thumbnail((400, 400))
            img_tk = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=img_tk, text="")
            self.preview_label.image = img_tk

            messagebox.showinfo("Success", "Face swapped successfully and saved as swapped_output.jpg")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Exception", str(e))

# Initialize app with safe CPU fallback
if __name__ == "__main__":
    try:
        model_path = ensure_model_exists()
        ctx_id = 0  # Use GPU if available
        try:
            face_app = FaceAnalysis(name='buffalo_l')
            face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        except:
            ctx_id = -1  # Fallback to CPU
            face_app = FaceAnalysis(name='buffalo_l')
            face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        swapper = get_model(model_path, download=False, download_zip=False)

        root = tk.Tk()
        app = FaceSwapperApp(root)
        root.mainloop()

    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)
