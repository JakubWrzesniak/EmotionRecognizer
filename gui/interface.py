import re
import tkinter as tk
from copy import copy
from tkinter import font as tkfont, END, NSEW, EW, W, NW
import threading
from Model.model import Emotion_model, my_models, model_list, EPOCH_HISTORY_PATH, CONFUSION_MATRIX_PATH, \
    PERFORMANCE_DIST_PATH, delete_model
from config.early_stopping import early_stopping
from config.lr_schedulers import lr_schedulers
from config.traind_datagen import train_datagen
from emotion_recognizer import cam_emo_rec
from gui.gui_template import Base_Gui
from gui.show_img import show_img

emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']


class EmotionRecognizer(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Comic', size=36, weight="bold")
        self.button_font = tkfont.Font(family='Comic', size=16)
        self.main_font = tkfont.Font(family='Comic', size=12)
        self.geometry_base = "1000x800+50+50"

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (Main_page, Create_model, Load_model):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("Main_page")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.refresh()
        frame.tkraise()

    def open_edit_window(self, edit_window):
        root = tk.Tk()
        app = self.windows[edit_window](master=root)
        app.mainloop()

    def refresh_all(self):
        for k, v in self.frames.items():
            v.refresh()


class Main_page(Base_Gui):

    def create_additive(self):
        self.frame = tk.Frame(self, bg="blue")
        self.title = tk.Label(self, text='Welcom in \nEmotion Recognizer', font=self.controller.title_font)
        self.create_model = tk.Button(self.frame, text="Create new model", font=self.controller.button_font,
                                      command=lambda: self.controller.show_frame('Create_model'))
        self.load_model = tk.Button(self.frame, text="Load model", font=self.controller.button_font,
                                    command=lambda: self.controller.show_frame('Load_model'))
        self.frame.grid(row=1, column=0)
        self.title.grid(row=0, column=0, sticky=NSEW)
        self.create_model.grid(row=0, column=0, sticky=EW)
        self.load_model.grid(row=1, column=0, sticky=EW)


class Create_model(Base_Gui):

    def remove(self, key, collection):
        if key in collection.keys():
            res = tk.messagebox.askyesno(title='confirmation', message='Are you sure that you want to delete {}?'
                                         .format(key))
            if res:
                del collection[key]
                self.refresh()

    def refresh(self):
        nltd = train_datagen
        nlrs = lr_schedulers
        nles = early_stopping
        if nltd != self.tdl:
            self.ltd = nltd
            self.list_train_datagen.delete(0, tk.END)
            for elem in self.ltd:
                self.list_train_datagen.insert(END, elem)
        if nlrs != self.lsl:
            self.lsl = nlrs
            self.list_lr_scheduler.delete(0, tk.END)
            for elem in self.lsl:
                self.list_lr_scheduler.insert(END, elem)
        if nles != self.esl:
            self.esl = nles
            self.list_early_stopping.delete(0, tk.END)
            for elem in self.esl:
                self.list_early_stopping.insert(END, elem)

    def learn(self, model):
        model.learn()

    def start_learning(self, model):
        import gui.loading as lo
        th = threading.Thread(target=self.learn, args=(model,))
        self.controller.withdraw()
        lo.loading(th)
        try:
            model.show_validation_metric()
            model.evaluate()
            tk.messagebox.showinfo(title="Success", message="Your model is created!")
        except:
            tk.messagebox.showwarning(title="Warnning", message="Model was not created!")
        self.controller.deiconify()
        self.controller.show_frame('Load_model')

    def submit(self):
        emotions = [i for i in range(len(self.emotion_var)) if self.emotion_var[i].get()]
        pattern = re.compile(r"[^a-zA-Z0-9_]+")
        if len(emotions) < 2:
            tk.messagebox.showwarning(title="Warnning", message="Please select at least two emotions")
        elif self.input_name.get() == "":
            tk.messagebox.showwarning(title="Warnning", message="Please input name")
        elif pattern.match(self.input_name.get()):
            tk.messagebox.showwarning(title="Warnning", message="Name can conatin onlu letters, numbers nad \"_\"")
        elif self.input_name.get() in model_list():
            tk.messagebox.showwarning(title="Warnning", message="Model with this name already exist!")
        elif not self.lr_scheduler_var:
            tk.messagebox.showwarning(title="Warnning", message="Select Lr scheduler")
        elif not self.trained_datagen_var:
            tk.messagebox.showwarning(title="Warnning", message="Select Train data gen")
        elif not self.early_stopping_var:
            tk.messagebox.showwarning(title="Warnning", message="Select Early stopping")
        else:
            new_model = Emotion_model.create_model(model_name=self.input_name.get(),
                                                   selected_emotions=emotions,
                                                   bach_size=self.batch_var.get(),
                                                   epochs=self.epoch_var.get(),
                                                   lr_scheduler=lr_schedulers[self.lr_scheduler_var],
                                                   early_stopping=early_stopping[self.early_stopping_var],
                                                   train_datagen=train_datagen[self.trained_datagen_var])

            self.start_learning(new_model)

    def callback_train(self, event):
        selection = event.widget.curselection()
        if selection:
            selection = self.list_train_datagen.get(selection)
            self.trained_datagen_var = selection
            data = train_datagen[selection]
            text = "\n".join([": ".join([name, str(val)]) for name, val in data.items()])
            self.label_trained_datagen.configure(text=text)

    def callback_early(self, event):
        selection = event.widget.curselection()
        if selection:
            selection = self.list_early_stopping.get(selection)
            self.early_stopping_var = selection
            data = early_stopping[selection]
            text = "\n".join([": ".join([name, str(val)]) for name, val in data.items()])
            self.label_early_stopping.configure(text=text)

    def callback_lr(self, event):
        selection = event.widget.curselection()
        if selection:
            selection = self.list_lr_scheduler.get(selection)
            self.lr_scheduler_var = selection
            data = lr_schedulers[selection]
            text = "\n".join([": ".join([name, str(val)]) for name, val in data.items()])
            self.label_lr_scheduler.configure(text=text)

    def create_additive(self):
        frame_main = tk.Frame(self)
        frame_emotions = tk.Frame(frame_main)
        frame_buttons = tk.Frame(frame_main)
        frame_parameters = tk.Frame(frame_main)
        label_title = tk.Label(self, text="Creat new model", font=self.controller.title_font)

        button_back = tk.Button(frame_buttons, text="Go to the start page",
                                command=lambda: self.controller.show_frame("Main_page"), fg='blue',
                                font=self.controller.button_font)

        button_submit = tk.Button(frame_buttons, text="Create model", command=self.submit, font=self.controller.button_font,
                                  fg='green')

        frame_trained_datagen = tk.LabelFrame(frame_parameters, highlightbackground="black", text='Trained datagen')
        frame_early_stopping = tk.LabelFrame(frame_parameters, highlightbackground="black", text='Early stopping')
        frame_lr_scheduler = tk.LabelFrame(frame_parameters, highlightbackground="black", text='LR scheduler')

        frame_buttons_tr = tk.Frame(frame_trained_datagen)
        frame_buttons_er = tk.Frame(frame_early_stopping)
        frame_buttons_lr = tk.Frame(frame_lr_scheduler)

        button_add_trained_datagen = tk.Button(frame_buttons_tr, text='+', width=2, height=2, font='5', fg='green',
                                               command=self.create_new_datagen)
        button_add_early_stopping = tk.Button(frame_buttons_er, text='+', width=2, height=2, font='5', fg='green',
                                              command=self.create_new_early_stopping)
        button_add_lr_scheduler = tk.Button(frame_buttons_lr, text='+', width=2, height=2, font='5', fg='green',
                                            command=self.create_new_lr_scheduler)

        button_del_trained_datagen = tk.Button(frame_buttons_tr, text='-', width=2, height=2, font='5', fg='red',
                                               command=lambda: self.remove(self.trained_datagen_var, train_datagen))
        button_del_early_stopping = tk.Button(frame_buttons_er, text='-', width=2, height=2, font='5', fg='red',
                                              command=lambda: self.remove(self.early_stopping_var, early_stopping))
        button_del_lr_scheduler = tk.Button(frame_buttons_lr, text='-', width=2, height=2, font='5', fg='red',
                                            command=lambda: self.remove(self.lr_scheduler_var, lr_schedulers))

        label_name = tk.Label(frame_main, text="Model name", font=self.controller.main_font)
        label_emotions = tk.Label(frame_main, text="Select emotions", font=self.controller.main_font)
        label_epoch = tk.Label(frame_main, text="Number of epoch", font=self.controller.main_font)
        label_batch = tk.Label(frame_main, text="Bach size", font=self.controller.main_font)

        self.label_trained_datagen = tk.Label(frame_trained_datagen, width=20)
        self.label_early_stopping = tk.Label(frame_early_stopping, width=20)
        self.label_lr_scheduler = tk.Label(frame_lr_scheduler, width=20)

        self.emotion_var = [tk.IntVar() for _ in range(len(emotions))]
        self.epoch_var = tk.IntVar()
        self.batch_var = tk.IntVar()
        self.dataGen_var = tk.IntVar()
        self.stopping_var = tk.IntVar()
        self.scheduler_var = tk.IntVar()
        self.trained_datagen_var = None
        self.early_stopping_var = None
        self.lr_scheduler_var = None

        self.input_name = tk.Entry(frame_main)
        emotion_checkButton = [
            tk.Checkbutton(frame_emotions, text=emotions[i], variable=self.emotion_var[i], onvalue=1, offvalue=0)
            for i in range(len(emotions))]

        scale_epoch = tk.Scale(frame_main, from_=1, to=500, orient=tk.HORIZONTAL, variable=self.epoch_var)
        scale_batch = tk.Scale(frame_main, from_=10, to=50, orient=tk.HORIZONTAL, variable=self.batch_var)

        self.list_train_datagen = tk.Listbox(frame_trained_datagen, font=self.controller.main_font, width=13)
        self.list_early_stopping = tk.Listbox(frame_early_stopping, font=self.controller.main_font, width=13)
        self.list_lr_scheduler = tk.Listbox(frame_lr_scheduler, font=self.controller.main_font, width=13)
        scrollbar_train_datagen = tk.Scrollbar(frame_trained_datagen, orient='vertical')
        scrollbar_train_datagen.config(command=self.list_train_datagen.yview)

        scrollbar_early_stopping = tk.Scrollbar(frame_early_stopping, orient='vertical')
        scrollbar_early_stopping.config(command=self.list_early_stopping.yview)

        scrollbar_lr_scheduler = tk.Scrollbar(frame_lr_scheduler, orient='vertical')
        scrollbar_lr_scheduler.config(command=self.list_lr_scheduler.yview)

        self.list_train_datagen.config(yscrollcommand=scrollbar_train_datagen.set)
        self.list_early_stopping.config(yscrollcommand=scrollbar_early_stopping.set)
        self.list_lr_scheduler.config(yscrollcommand=scrollbar_lr_scheduler.set)

        self.tdl = copy(train_datagen)
        self.esl = copy(early_stopping)
        self.lsl = copy(lr_schedulers)

        for k in self.tdl.keys():
            self.list_train_datagen.insert(END, k)

        for k in self.esl.keys():
            self.list_early_stopping.insert(END, k)

        for k in self.lsl.keys():
            self.list_lr_scheduler.insert(END, k)

        self.batch_var.set(32)
        self.epoch_var.set(50)

        frame_main.grid(row=1, column=0)
        frame_emotions.grid(row=2, column=1)
        frame_parameters.grid(row=5, padx=20, pady=20, columnspan=2)
        frame_buttons.grid(row=7, column=1)
        label_title.grid(row=0, ipadx=20, ipady=20, sticky=NSEW)
        label_name.grid(row=1, column=0, ipadx=20, ipady=5, sticky=W)
        self.input_name.grid(row=1, column=1, ipadx=20, ipady=5, sticky=W)
        label_emotions.grid(row=2, column=0, ipadx=20, ipady=5, sticky=W)
        label_epoch.grid(row=3, column=0, ipadx=20, ipady=5, sticky=W)
        label_batch.grid(row=4, column=0, ipadx=20, ipady=5, sticky=W)
        frame_buttons_lr.pack(side='bottom', fill="y")
        frame_buttons_er.pack(side='bottom', fill="y")
        frame_buttons_tr.pack(side='bottom', fill="y")
        self.list_train_datagen.pack(side="left", fill="y")
        self.list_early_stopping.pack(side="left", fill="y")
        self.list_lr_scheduler.pack(side="left", fill="y")
        self.label_trained_datagen.pack(side="right", fill="y")
        self.label_early_stopping.pack(side="right", fill="y")
        self.label_lr_scheduler.pack(side="right", fill="y")
        scrollbar_train_datagen.pack(side="right", fill="y")
        scrollbar_early_stopping.pack(side="right", fill="y")
        scrollbar_lr_scheduler.pack(side="right", fill="y")
        frame_trained_datagen.grid(row=0, column=1, sticky=NSEW)
        frame_early_stopping.grid(row=0, column=2, sticky=NSEW)
        frame_lr_scheduler.grid(row=0, column=3, sticky=NSEW)

        button_add_lr_scheduler.grid(row=0, column=0, sticky=EW)
        button_add_early_stopping.grid(row=0, column=0, sticky=EW)
        button_add_trained_datagen.grid(row=0, column=0, sticky=EW)
        button_del_lr_scheduler.grid(row=0, column=1, sticky=EW)
        button_del_early_stopping.grid(row=0, column=1, sticky=EW)
        button_del_trained_datagen.grid(row=0, column=1, sticky=EW)

        for i in range(len(emotion_checkButton)):
            emotion_checkButton[i].grid(row=0, column=i, sticky=W)
        scale_epoch.grid(row=3, column=1, sticky=EW)
        scale_batch.grid(row=4, column=1, sticky=EW)

        button_submit.grid(row=0, column=0, ipady=5, pady=20, padx=5, sticky=EW)
        button_back.grid(row=0, column=1, ipady=5, pady=20, padx=5, sticky=EW)

        self.list_train_datagen.bind("<<ListboxSelect>>", self.callback_train)
        self.list_early_stopping.bind("<<ListboxSelect>>", self.callback_early)
        self.list_lr_scheduler.bind("<<ListboxSelect>>", self.callback_lr)


class Load_model(Base_Gui):

    def show_epoch(self):
        try:
            md = self.model_list[self.index]
            show_img(md + " Epoch history", EPOCH_HISTORY_PATH + md + ".png")
        except:
            tk.messagebox.showwarning(title='Warning', message="Select model first")

    def show_matrix(self):
        try:
            md = self.model_list[self.index]
            show_img(md + " Confusion matrix", CONFUSION_MATRIX_PATH + md + ".png")
        except:
            tk.messagebox.showwarning(title='Warning', message="Select model first")

    def show_performance(self):
        try:
            md = self.model_list[self.index]
            show_img(md + " Performance distance", PERFORMANCE_DIST_PATH + md + ".png")
        except:
            tk.messagebox.showwarning(title='Warning', message="Select model first")

    def delete(self):
        if self.model_list[self.index]:
            res = tk.messagebox.askyesno(title='confirmation', message='Are you sure that you want to remove model?')
        if res and delete_model(self.model_list[self.index]):
            tk.messagebox.showinfo(title='Succes', message='Model is removed', icon='info')
            self.refresh()
        else:
            tk.messagebox.showerror(title='Succes', message='Can not remove model', icon='error')

    def refresh(self):
        new_model_list = model_list()
        if new_model_list != self.model_list:
            self.model_list = new_model_list
            self.my_models = my_models()
            self.list_models.delete(0, tk.END)
            for elem in self.model_list:
                self.list_models.insert(END, elem)

    def callback(self, event):

        def load_emotions():
            data = self.my_models[self.index].model_emotions
            to_text = "\n".join(data)
            self.label_emotions.configure(text=to_text)

        def load_train_datagen():
            data = self.my_models[self.index].train_datagen_pattern
            to_text = "\n".join([": ".join([name, str(val)]) for name, val in data.items()])
            self.label_trainde_dategen.configure(text=to_text)

        def load_early_stopping():
            data = self.my_models[self.index].early_stopping_pattern
            to_text = "\n".join([": ".join([name, str(val)]) for name, val in data.items()])
            self.label_early_stopping.configure(text=to_text)

        def load_lr_scheduler():
            data = self.my_models[self.index].lr_scheduler_pattern
            to_text = "\n".join([": ".join([name, str(val)]) for name, val in data.items()])
            self.label_lr_scheduler.configure(text=to_text)

        def load_epoch_bach():
            md = self.my_models[self.index]
            to_text = ""
            to_text += "Batch size: " + str(md.batch_size) + "\n"
            to_text += "Epoch: " + str(md.history_epoch[-1] + 1)
            self.label_batch_epoch.configure(text=to_text)

        selection = event.widget.curselection()
        if selection:
            self.index = selection[0]
            load_emotions()
            load_train_datagen()
            load_early_stopping()
            load_lr_scheduler()
            load_epoch_bach()
        else:
            self.label_trainde_dategen.configure(text="")

    def select(self):
        try:
            cam_emo_rec.run(self.my_models[self.index])
        except AttributeError as e:
            tk.messagebox.showwarning(title='Warning', message="Select model first")

    def create_additive(self):

        frame_main = tk.Frame(self)
        frame_edit_buttons = tk.Label(frame_main)
        label = tk.Label(self, text="Select model", font=self.controller.title_font)
        self.list_models = tk.Listbox(frame_main, height=17, font=self.controller.main_font)
        button_back = tk.Button(frame_main, text="Go to the start page",
                                command=lambda: self.controller.show_frame("Main_page"), fg='blue',
                                font=self.controller.button_font)
        button_select = tk.Button(frame_main, text="Select model", command=self.select, font=self.controller.button_font,
                                  fg='green', )
        button_matrix = tk.Button(frame_main, text="Confusion Matrix", command=self.show_matrix,
                                  font=self.controller.main_font)
        button_epoch = tk.Button(frame_main, text="Epoch history", command=self.show_epoch,
                                 font=self.controller.main_font)
        button_dist = tk.Button(frame_main, text="Performance_dist", command=self.show_performance,
                                font=self.controller.main_font)
        button_add_model = tk.Button(frame_edit_buttons, text='+', width=3, height=3, font='10', fg='green',
                                     command=lambda: self.controller.show_frame('Create_model'))

        button_del_model = tk.Button(frame_edit_buttons, text='-', width=3, height=3, font='10', fg='red',
                                     command=lambda: self.delete())

        frame_trained_datagen = tk.LabelFrame(frame_main, highlightbackground="black", text='Trained datagen')
        frame_early_stopping = tk.LabelFrame(frame_main, highlightbackground="black", text='Early stopping')
        frame_lr_scheduler = tk.LabelFrame(frame_main, highlightbackground="black", text='LR scheduler')
        frame_emotions = tk.LabelFrame(frame_main, highlightbackground="black", text='Emotions')
        frame_batch_epoch = tk.LabelFrame(frame_main, highlightbackground="black", text='Epoch/Bach')

        self.label_trainde_dategen = tk.Label(frame_trained_datagen, width=20, height=10,
                                              font=self.controller.main_font)
        self.label_early_stopping = tk.Label(frame_early_stopping, width=20, height=10, font=self.controller.main_font)
        self.label_lr_scheduler = tk.Label(frame_lr_scheduler, width=20, height=10, font=self.controller.main_font)
        self.label_emotions = tk.Label(frame_emotions, width=20, height=10, font=self.controller.main_font)
        self.label_batch_epoch = tk.Label(frame_batch_epoch, width=20, height=7, font=self.controller.main_font)

        frame_main.grid(row=1, column=0)

        label.grid(row=0, column=0, sticky=NSEW)
        self.list_models.grid(row=0, rowspan=2, column=0, sticky=W)
        button_select.grid(row=2, column=3, columnspan=2, ipady=5, pady=20, padx=5, sticky=EW)
        button_back.grid(row=2, column=1, columnspan=2, ipady=5, pady=20, padx=5, sticky=EW)
        button_epoch.grid(row=1, column=2, sticky=NSEW)
        button_matrix.grid(row=1, column=3, sticky=NSEW)
        button_dist.grid(row=1, column=4, sticky=NSEW)
        button_add_model.grid(row=0, column=0, sticky=W)
        button_del_model.grid(row=0, column=1, sticky=W)
        self.label_emotions.grid(sticky=W)
        self.label_trainde_dategen.grid(sticky=W)
        self.label_early_stopping.grid(sticky=W)
        self.label_lr_scheduler.grid(sticky=W)
        self.label_batch_epoch.grid(sticky=W)

        self.model_list = model_list()
        self.my_models = my_models()

        for elem in self.model_list:
            self.list_models.insert(END, elem)

        frame_emotions.grid(row=0, column=1, sticky=NW)
        frame_trained_datagen.grid(row=0, column=2, sticky=NW)
        frame_early_stopping.grid(row=0, column=3, sticky=NW)
        frame_lr_scheduler.grid(row=0, column=4, sticky=NW)
        frame_batch_epoch.grid(row=1, column=1, sticky=NW)
        frame_edit_buttons.grid(row=2, column=0)

        self.list_models.bind("<<ListboxSelect>>", self.callback)
