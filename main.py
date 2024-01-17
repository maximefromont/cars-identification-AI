import cv2
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog
import numpy as np


class PhotoApp(ctk.CTkFrame):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self.master.title("AI Car model recognition - Kévin Martin Maxime - 2024")

        self.canvas = ctk.CTkCanvas(self, width=400, height=400)
        self.canvas.configure(bg=self.master.cget("bg"), highlightthickness=0) 
        self.canvas.pack(side=ctk.TOP)
        

        self.resultAI = ctk.CTkLabel(self, text="")
        self.resultAI.pack(side=ctk.TOP)

        self.button_open = ctk.CTkButton(self, text="Ouvrir", command=self.open_file)
        self.button_open.pack(side=ctk.TOP)

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])

        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    self.display_image(image)
                    # Mettre ici l'appel vers l'IA
                    self.resultAI.configure(text="Chemin du fichier : " + file_path) # Mettre ici le texte avec le résultat de l'IA
                    self.canvas.pack(fill=ctk.BOTH, expand=True)
                else:
                    self.resultAI.configure(text="Erreur : Impossible de lire l'image") # Default en cas de problème
                    self.canvas.delete("all")
            except Exception as e:
                self.resultAI.configure(text="Erreur : " + str(e)) # Pbl pour lire le fichier

    def display_image(self, image):
        self.canvas.delete("all")

        image.thumbnail((400, 400))

        tk_image = ImageTk.PhotoImage(image)

        self.canvas.image = tk_image
        self.canvas.create_image(0, 0, anchor=ctk.NW, image=tk_image)

if __name__ == "__main__":
    root = ctk.CTk()
    app = PhotoApp(root)
    app.pack()
    root.mainloop()