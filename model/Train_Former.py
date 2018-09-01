import os
import json

class Train_Former:
    def __init__(self,
                 save_dir =
                 "/home/wooden/Desktop/Pycharm_projects/Flask_Keras"
                 "/static/Emnist_dir/Own_classes/save",
                 train_dir =
                 "/home/wooden/Desktop/Pycharm_projects/Flask_Keras"
                 "/static/Emnist_dir/Own_classes/train",
                 uppercase = "/uppercase",
                 lowercase = "/lowercase",
                 numbers = "numbers",
                 Classifajar = "Classifajar"
                 ):
        self.save_dir = save_dir
        self.train_dir = train_dir
        self.uppercase = uppercase
        self.lowercase = lowercase
        self.numbers = numbers
        self.Classifajar = Classifajar
        with open("TrFo_Self.json") as f:
            file = json.load(f)
        self.Json_self = file

    @property
    def Json_Self(self):
        with open("TrFo_Self.json") as f:
            file = json.load(f)
        Json_Self = file
        return Json_Self

    def count_dir(self, url):
        dir_count = len(next(os.walk(url))[1])
        return dir_count
    def file_count(self, url):
        file_count = len(next(os.walk(url))[2])
        return file_count




