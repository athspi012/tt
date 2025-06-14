import os
import sys
import urllib.request
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper

MODEL_NAME = "inswapper_128.onnx"
MODEL_URL = f"https://huggingface.co/henryruhs/insightface-models/resolve/main/{MODEL_NAME}"

# Determine path for resources (works in .exe)
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Download model if missing, with user notification
def ensure_model():
    model_path = resource_path(MODEL_NAME)
    if not os.path.exists(model_path):
        messagebox.showinfo("Setup", "Installing AI model files... This happens only once.")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
        except Exception as e:
            messagebox.showerror("Download Error", f"Failed to download model:\n{e}")
            sys.exit(1)
    return model_path

class FaceSwapperApp:
    def __init__(self, root, face_app, swapper):
        self.root = root
        self.face_app = face_app
        self.swapper = swapper

        root.title("Peodecsob AI Face Swapper")
        root.geometry("920x740")
        root.configure(bg="#f5f5f5")

        self.src_path = ""
        self.tgt_path = ""

        tk.Label(root, text="Peodecsob Face Swapper", font=("Arial", 22, "bold"), bg="#f5f5f5").pack(pady=15)
        tk.Button(root, text="📤 Load Source Face", command=self.load_source, font=("Arial", 14), width=40).pack(pady=10)
        tk.Button(root, text="🖼 Load Target Image", command=self.load_target, font=("Arial", 14), width=40).pack(pady=10)
        tk.Button(root, text="🔁 Swap Faces", command=self.swap_faces, font=("Arial", 14), bg="#007ACC", fg="white", width=40).pack(pady=20)

        self.preview = tk.Label(root, text="Preview appears here", bg="#f5f5f5")
        self.preview.pack(pady=15)

    def load_source(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if path:
            self.src_path = path
            messagebox.showinfo("Source Selected", path)

    def load_target(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if path:
            self.tgt_path = path
            messagebox.showinfo("Target Selected", path)

    def swap_faces(self):
        try:
            if not self.src_path or not self.tgt_path:
                messagebox.showwarning("Missing Images", "Please load both source and target images.")
                return

            src = cv2.imread(self.src_path)
            tgt = cv2.imread(self.tgt_path)
            src_faces = self.face_app.get(src)
            tgt_faces = self.face_app.get(tgt)

            if not src_faces or not tgt_faces:
                messagebox.showerror("Detection Failed", "Faces not detected clearly.")
                return

            result = self.swapper.get(tgt, tgt_faces[0], src_faces[0], paste_back=True)
            out = "swapped_output.jpg"
            cv2.imwrite(out, result)

            img = Image.open(out)
            img.thumbnail((450, 450))
            photo = ImageTk.PhotoImage(img)
            self.preview.configure(image=photo)
            self.preview.image = photo

            messagebox.showinfo("Success", f"Saved to {out}")
        except Exception:
            with open("error_log.txt","w") as f:
                f.write(traceback.format_exc())
            messagebox.showerror("Error", "An error occurred. See error_log.txt")

if __name__ == '__main__':
    try:
        model_path = ensure_model()
        try:
            fa = FaceAnalysis(name="buffalo_l")
            fa.prepare(ctx_id=0, det_size=(640,640))
            ctx = 0
        except:
            fa = FaceAnalysis(name="buffalo_l")
            fa.prepare(ctx_id=-1, det_size=(640,640))
            ctx = -1
        sw = INSwapper(model_path, ctx_id=ctx)
        root = tk.Tk()
        FaceSwapperApp(root, fa, sw)
        root.mainloop()
    except Exception:
        with open("fatal_error_log.txt","w") as f:
            f.write(traceback.format_exc())
        print("Fatal error. Check fatal_error_log.txt")
