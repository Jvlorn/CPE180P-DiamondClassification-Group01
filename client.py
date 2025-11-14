# client_tkinter.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import requests
import os

class DiamondClassifierClient:
    def __init__(self, server_url="http://127.0.0.1:8000"):
        self.server_url = server_url
        self.is_connected = False

    def test_connection(self):
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.is_connected = data.get("model_loaded", False)
                return True, "Connected to server - Model ready" if self.is_connected else "Connected but model not loaded"
            return False, f"Server error: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Cannot connect to server: {e}"

    def predict_image(self, image_path):
        try:
            with open(image_path, "rb") as f:
                files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
                response = requests.post(f"{self.server_url}/predict", files=files, timeout=30)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"Prediction failed: {response.text}"
        except Exception as e:
            return False, f"Error: {e}"

class DiamondClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diamond Quality Classifier")
        self.client = DiamondClassifierClient()
        self.selected_file_path = None

        # Server URL
        tk.Label(root, text="Server URL:").grid(row=0, column=0, sticky="w")
        self.server_url_entry = tk.Entry(root, width=50)
        self.server_url_entry.insert(0, "http://127.0.0.1:8000")
        self.server_url_entry.grid(row=0, column=1)
        tk.Button(root, text="Connect", command=self.connect_to_server).grid(row=0, column=2)

        self.connection_status = tk.Label(root, text="Not connected", fg="red")
        self.connection_status.grid(row=1, column=0, columnspan=3, sticky="w")

        # File selection
        tk.Button(root, text="📁 Choose Image", command=self.choose_file).grid(row=2, column=0)
        self.selected_file_label = tk.Label(root, text="No file selected")
        self.selected_file_label.grid(row=2, column=1, columnspan=2, sticky="w")

        # Image preview
        self.image_label = tk.Label(root)
        self.image_label.grid(row=3, column=0, rowspan=4, padx=10, pady=10)

        # Results
        self.result_text = tk.Text(root, width=50, height=15)
        self.result_text.grid(row=3, column=1, columnspan=2, padx=10, pady=10)

        # Classify button
        tk.Button(root, text="🔍 Classify Diamond", command=self.classify_image).grid(row=7, column=1, columnspan=2)

    def connect_to_server(self):
        self.client.server_url = self.server_url_entry.get()
        success, message = self.client.test_connection()
        self.connection_status.config(text=message, fg="green" if success else "red")

    def choose_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.selected_file_path = file_path
            self.selected_file_label.config(text=os.path.basename(file_path))
            self.show_image_preview(file_path)
        else:
            self.selected_file_label.config(text="No file selected")
            self.selected_file_path = None
            self.image_label.config(image="")

    def show_image_preview(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((300, 300))
            self.tk_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_image)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load image: {e}")

    def classify_image(self):
        if not self.client.is_connected:
            messagebox.showwarning("Warning", "Not connected to server")
            return
        if not self.selected_file_path or not os.path.exists(self.selected_file_path):
            messagebox.showwarning("Warning", "Please select a valid image file")
            return

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Predicting...\n")
        self.root.update()

        success, result = self.client.predict_image(self.selected_file_path)
        self.result_text.delete("1.0", tk.END)

        if success:
            self.display_prediction_results(result)
        else:
            self.result_text.insert(tk.END, f"❌ Error: {result}")

    def display_prediction_results(self, result):
        predictions = result.get("predictions", {})
        if not predictions:
            self.result_text.insert(tk.END, "No prediction data returned.\n")
            return

        avg_conf = sum(v['confidence'] for v in predictions.values()) / len(predictions)
        self.result_text.insert(tk.END, f"Overall Confidence: {avg_conf:.1%}\n\n")

        for attribute, pred in predictions.items():
            self.result_text.insert(tk.END, f"{attribute}: {pred['label']} ({pred['confidence']:.1%})\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = DiamondClassifierApp(root)
    root.mainloop()
