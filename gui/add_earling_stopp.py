from tkinter import NSEW, W, E, EW, SW
import tkinter as tk


from config.early_stopping import early_stopping

MONITOR_VAR = ['val_accuracy', 'val_loss', 'accuracy', 'loss']


class Add_earling_stop(tk.Frame):

    def __init__(self, controller, master=None):
        tk.Frame.__init__(self, master)
        self.geometria_baza = "600x400+150+150"
        self.controller = controller
        self.parent = master
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
            if self.input_name.get() in early_stopping.keys():
                reply = tk.messagebox.askyesno(
                    "Item exist",
                    "Item with this name exist.\nWould you like overwrite it?", parent=self.parent, type=tk.messagebox.YESNO)
            if reply:
                val = {
                    "monitor": self.monitor_var.get(),
                    "min_delta": self.mid_delta_var.get(),
                    "patience": self.patience_var.get(),
                    "verbose": self.verbose_var.get(),
                    "restore_best_weights": self.best_weight_var.get()
                }

                early_stopping[self.input_name.get()] = val
                self.controller.refresh_all()
                self.parent.destroy()

    def create_view(self):
        self.label_name = tk.Label(self.parent, text="Name")
        self.label_monitor = tk.Label(self.parent, text="Monitor")
        self.label_min_delta = tk.Label(self.parent, text="Min_delta")
        self.label_patience = tk.Label(self.parent, text="Patience")
        self.label_verbose = tk.Label(self.parent, text="Verbose")
        self.label_best_weights = tk.Label(self.parent, text="Restore best weights")

        self.frame_checkbox = tk.Frame(self.parent)

        self.monitor_var = tk.StringVar()
        self.mid_delta_var = tk.DoubleVar()
        self.patience_var = tk.IntVar()
        self.verbose_var = tk.IntVar()
        self.best_weight_var = tk.BooleanVar()

        self.monitor_var.set(MONITOR_VAR[0])

        self.input_name = tk.Entry(self.parent)
        self.radio_monitor = [tk.Radiobutton(self.frame_checkbox, text=MONITOR_VAR[i], variable=self.monitor_var, value=MONITOR_VAR[i])
                          for i in range(len(MONITOR_VAR))]
        self.scale_mid_delta = tk.Scale(self.parent, from_=0., to=0.0009, resolution=0.0001, orient=tk.HORIZONTAL,
                                        variable=self.mid_delta_var)
        self.scale_patience = tk.Scale(self.parent, from_=1, to=10, resolution=1, orient=tk.HORIZONTAL,
                                     variable=self.patience_var)
        self.scale_verbose = tk.Scale(self.parent, from_=0, to=1, resolution=1, orient=tk.HORIZONTAL,
                                    variable=self.verbose_var)
        self.check_best_weight = tk.Checkbutton(self.parent, variable=self.best_weight_var, onvalue=True, offvalue=False)

        self.button_submit = tk.Button(self.parent, text="Save", command=self.submit)

        self.label_name.grid(row=0, column=0, padx=20, sticky=W)
        self.label_monitor.grid(row=1, column=0, padx=20, sticky=W)
        self.label_min_delta.grid(row=2, column=0, padx=20, sticky=W)
        self.label_patience.grid(row=3, column=0, padx=20, sticky=W)
        self.label_verbose.grid(row=4, column=0, padx=20, sticky=W)
        self.label_best_weights.grid(row=5, column=0, padx=20, sticky=W)

        self.frame_checkbox.grid(row=1, column=1, padx=20, sticky=EW)

        self.input_name.grid(row=0, column=1, padx=20, sticky=EW)
        for i in range(len(self.radio_monitor)):
            self.radio_monitor[i].pack(side="left")
        self.scale_mid_delta.grid(row=2, column=1, padx=20, sticky=EW)
        self.scale_patience.grid(row=3, column=1, padx=20, sticky=EW)
        self.scale_verbose.grid(row=4, column=1, padx=20, sticky=EW)
        self.check_best_weight.grid(row=5, column=1, padx=20, sticky=EW)
        self.button_submit.grid(row=7, column=1, pady=20, sticky=EW)

    def file_quit(self):
        reply = tk.messagebox.askyesno(
            "exit",
            "If you exit you will lose your data?", parent=self.parent)
        if reply:
            self.parent.destroy()
