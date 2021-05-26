import tkinter as tk
from tkinter import NSEW, W, E, EW, SW

from config.traind_datagen import train_datagen


class Add_trained_datagen(tk.Frame):

    def __init__(self, controller, master=None):
        tk.Frame.__init__(self, master)
        self.geometria_baza = "600x400+150+150"
        self.parent = master
        self.controller = controller
        self.parent.protocol("WM_DELETE_WINDOW", self.file_quit)
        self.parent.geometry(self.geometria_baza)
        self.parent.columnconfigure(0, weight=1)
        self.parent.columnconfigure(1, weight=10)
        self.parent.rowconfigure(0, weight=1)
        self.parent.rowconfigure(1, weight=1)
        self.parent.rowconfigure(2, weight=1)
        self.parent.rowconfigure(3, weight=1)
        self.parent.rowconfigure(4, weight=1)
        self.parent.rowconfigure(5, weight=1)
        self.parent.rowconfigure(6, weight=1)
        self.create_view()

    def submit(self):
        reply = True
        if len(self.input_name.get()) == 0:
            tk.messagebox.showwarning(title="Error", message="You need to enter the name", type=tk.messagebox.OK)
        else:
            if self.input_name.get() in train_datagen.keys():
                reply = tk.messagebox.askyesno(
                    "Item exist",
                    "Item with this name exist.\nWould you like overwrite it?", parent=self.parent,
                    type=tk.messagebox.YESNO)
            if reply:
                val = {
                    "rotation_range": self.rotation_var.get(),
                    "width_shift_range": self.width_var.get(),
                    "height_shift_range": self.height_var.get(),
                    "shear_range": self.share_var.get(),
                    "zoom_range": self.zoom_var.get(),
                    "horizontal_flip": self.horizontal_var.get()
                }

                train_datagen[self.input_name.get()] = val
                self.controller.refresh_all()
                self.parent.destroy()

    def create_view(self):
        self.label_name = tk.Label(self.parent, text="Name")
        self.label_roataion = tk.Label(self.parent, text="Rotation range")
        self.label_width = tk.Label(self.parent, text="Width shift range")
        self.label_height = tk.Label(self.parent, text="Height shift range")
        self.label_share = tk.Label(self.parent, text="Shear range")
        self.label_zoom = tk.Label(self.parent, text="Zoom range")
        self.label_flip = tk.Label(self.parent, text="Horizontal_flip")

        self.rotation_var = tk.IntVar()
        self.width_var = tk.DoubleVar()
        self.height_var = tk.DoubleVar()
        self.share_var = tk.DoubleVar()
        self.zoom_var = tk.DoubleVar()
        self.horizontal_var = tk.BooleanVar()

        self.input_name = tk.Entry(self.parent)
        self.scale_rotation = tk.Scale(self.parent, from_=0, to=10, resolution=1, length=200, orient=tk.HORIZONTAL, variable=self.rotation_var)
        self.scale_width = tk.Scale(self.parent, from_=-1., to=0.9999, resolution=0.05, length=200, orient=tk.HORIZONTAL, variable=self.width_var)
        self.scale_height = tk.Scale(self.parent, from_=-1., to=0.9999, resolution=0.05, length=200, orient=tk.HORIZONTAL, variable=self.height_var)
        self.scale_share = tk.Scale(self.parent, from_=0., to=0.9999, resolution=0.05, length=200, orient=tk.HORIZONTAL, variable=self.share_var)
        self.scale_zoom = tk.Scale(self.parent, from_=0., to=0.15, resolution=0.01, length=200, orient=tk.HORIZONTAL, variable=self.zoom_var)
        self.check_horizontal = tk.Checkbutton(self.parent, variable=self.horizontal_var, onvalue=True, offvalue=False)

        self.button_submit = tk.Button(self.parent, text="Save", command=self.submit)

        self.label_name.grid(row=0, column=0, padx=20, sticky=SW)
        self.label_roataion.grid(row=1, column=0, padx=20, sticky=SW)
        self.label_width.grid(row=2, column=0, padx=20, sticky=SW)
        self.label_height.grid(row=3, column=0, padx=20, sticky=SW)
        self.label_share.grid(row=4, column=0, padx=20, sticky=SW)
        self.label_zoom.grid(row=5, column=0, padx=20, sticky=SW)
        self.label_flip.grid(row=6, column=0, padx=20, sticky=SW)

        self.input_name.grid(row=0, column=1, padx=20, sticky=EW)
        self.scale_rotation.grid(row=1, column=1, padx=20, sticky=EW)
        self.scale_width.grid(row=2, column=1, padx=20, sticky=EW)
        self.scale_height.grid(row=3, column=1, padx=20, sticky=EW)
        self.scale_share.grid(row=4, column=1, padx=20, sticky=EW)
        self.scale_zoom.grid(row=5, column=1, padx=20, sticky=EW)
        self.check_horizontal.grid(row=6, column=1, padx=20, sticky=EW)
        self.button_submit.grid(row=7, column=1, padx=20, pady=20, sticky=EW)

    def file_quit(self):
        reply = tk.messagebox.askyesno(
            "exit",
            "If you exit you will lose your data?", parent=self.parent)
        if reply:
            self.parent.destroy()

