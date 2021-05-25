'''
Created on 2021-03-22

@author: Andrzej
'''

import configparser
import tkinter as tk
from PIL import Image, ImageTk
import tkinter.messagebox
from tkinter.constants import NSEW, W, EW
import os

import config.traind_datagen
import config.lr_schedulers
import config.early_stopping
from gui.add_earling_stopp import Add_earling_stop
from gui.add_lr_scheduler import Add_lr_scheduler
from gui.add_trained_datagen import Add_trained_datagen

dane_konfig = "c:/Python/m_config.txt"


class BazoweGui(tk.Frame):
    def __init__(self, parent, controller):
        self.controller = controller
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.geometria_baza = "1200x800+50+50"
        self.controller.geometry(self.geometria_baza)
        self.controller.protocol("WM_DELETE_WINDOW", self.file_quit)
        self.utworz_bazowe_menu()
        self.dodaj_menu_custom()
        self.dodaj_menu_help()
        self.utworz_dodatki()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=999999)
        self.rowconfigure(2, weight=1)

    def utworz_dodatki(self):
        pass

    def refresh(self):
        pass

    def utworz_bazowe_menu(self):
        self.menubar = tk.Menu(self.parent)
        self.controller["menu"] = self.menubar
        fileMenu = tk.Menu(self.menubar)
        for label, command, shortcut_text, shortcut in (
                ("New dataGen", lambda: self.create_new_datagen(), "Ctrl+D", "<Control-d>"),
                ("New early stoping", lambda: self.create_new_early_stopping(), "Ctrl+R", "<Control-r>"),
                ("New lr scheduler", lambda: self.create_new_lr_scheduler(), "Ctrl+L", "<Control-l>"),
                (None, None, None, None)
        ):
            if label is None:
                fileMenu.add_separator()
            else:
                fileMenu.add_command(label=label, underline=0,
                                     command=command, accelerator=shortcut_text)
                self.parent.bind(shortcut, command)
        self.menubar.add_cascade(label="File", menu=fileMenu, underline=0)
        pass

    def dodaj_menu_help(self):
        fileMenu = tk.Menu(self.menubar)
        for label, command, shortcut_text, shortcut in ():
            if label is None:
                fileMenu.add_separator()
            else:
                fileMenu.add_command(label=label, underline=0,
                                     command=command, accelerator=shortcut_text)
                self.parent.bind(shortcut, command)
        self.menubar.add_cascade(label="Help", menu=fileMenu, underline=0)
        pass

    def file_quit(self, event=None):
        reply = tkinter.messagebox.askyesno(
            "koniec pracy",
            "naprawdę kończysz?", parent=self.controller)
        event = event
        if reply:
            config.traind_datagen.save()
            config.lr_schedulers.save()
            config.early_stopping.save()
            self.controller.destroy()
        pass

    def dodaj_menu_custom(self):
        pass

    def create_new_datagen(self):
        root = tk.Toplevel(self.controller)
        app = Add_trained_datagen(self.controller, root)
        app.mainloop()

    def create_new_early_stopping(self):
        root = tk.Toplevel(self.controller)
        app = Add_earling_stop(self.controller, root)
        app.mainloop()

    def create_new_lr_scheduler(self):
        root = tk.Toplevel(self.controller)
        app = Add_lr_scheduler(self.controller, root)
        app.mainloop()

    def create_loading_view(self):
        pass
