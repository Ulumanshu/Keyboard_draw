import os
import re
import json

class Train_Former:
    def __init__(self,
                 save_dir =
                 "/home/wooden/Desktop/Pycharm_projects/Flask_Keras"
                 "/static/Own_classes/save",
                 train_dir =
                 "/home/wooden/Desktop/Pycharm_projects/Flask_Keras"
                 "/static/Own_classes/train",
                 uppercase="/uppercase",
                 lowercase="/lowercase",
                 numbers="/numbers",
                 classifajar="/Classifajar"
                 ):
        self.save_dir = save_dir
        self.train_dir = train_dir
        self.uppercase = uppercase
        self.lowercase = lowercase
        self.numbers = numbers
        self.classifajar = classifajar
        self.train_upper = self.train_dir + self.uppercase
        self.train_lower = self.train_dir + self.lowercase
        self.train_numbr = self.train_dir + self.numbers
        self.train_class = self.train_dir + self.classifajar
        self.save_upper = self.save_dir + self.uppercase
        self.save_lower = self.save_dir + self.lowercase
        self.save_numbr = self.save_dir + self.numbers

    @property
    def Json_Self(self):
        with open("TrFo_Self.json") as f:
            file = json.load(f)
        Json_Self = file
        return Json_Self

    def accountant(self):
        Json_Self = {}
        save_list = [self.save_upper, self.save_lower, self.save_numbr]
        train_list = [self.train_upper, self.train_lower,
                      self.train_numbr, self.train_class]
        Json_Self["Save_dir"] = {}
        Json_Self["Train_dir"] = {}
        for e in save_list:
            fcs_list = []
            dir_cnt, dir_ls = Train_Former.count_dir(e)
            dir_name_abr = re.search(r"([^/]+$)", str(e)).group(1)
            Json_Self["Save_dir"][dir_name_abr] = {}
            Json_Self["Save_dir"][dir_name_abr]["Dir_count"] = dir_cnt
            for i in dir_ls:
                file_cnt = Train_Former.count_file(e + "/" + i)
                fcs_list.append(file_cnt)
                min_value = min(fcs_list)
                total = sum(fcs_list)
                Json_Self["Save_dir"][dir_name_abr]["Min_fc"] = min_value
                Json_Self["Save_dir"][dir_name_abr]["Total_files"] = total
                Json_Self["Save_dir"][dir_name_abr][i] = file_cnt
        for e in train_list:
            fct_list = []
            dir_cnt, dir_ls = Train_Former.count_dir(e)
            dir_name_abr = re.search(r"([^/]+$)", str(e)).group(1)
            Json_Self["Train_dir"][dir_name_abr] = {}
            Json_Self["Train_dir"][dir_name_abr]["Dir_count"] = dir_cnt
            for i in dir_ls:
                file_cnt = Train_Former.count_file(e + "/" + i)
                fct_list.append(file_cnt)
                min_value = min(fct_list)
                total = sum(fct_list)
                Json_Self["Train_dir"][dir_name_abr]["Min_fc"] = min_value
                Json_Self["Train_dir"][dir_name_abr]["Total_files"] = total
                Json_Self["Train_dir"][dir_name_abr][i] = file_cnt
        with open("TrFo_Self.json", 'w') as f:
            json.dump(Json_Self, f, indent=4, sort_keys=True)
        return print(Json_Self)


    @staticmethod
    def read_file(dir_c, fname):
        with open(os.path.join(dir_c, fname), 'rb') as file:
            output = file.read()
        return output

    @staticmethod
    def save_file(dir_d, fname, file):
        if os.path.exists(dir_d) == False:
            os.makedirs(dir_d)
        with open(os.path.join(dir_d, fname), 'wb') as output:
            output.write(file)
        return print("File saved: {}/ {}".format(dir_d, fname))

    @staticmethod
    def count_dir(dir_d):
        dir_count = len(next(os.walk(dir_d))[1])
        dir_list = []
        for root, dirs, files in os.walk(dir_d):
            dir_list.append(dirs)

        return dir_count, dir_list[0]

    @staticmethod
    def count_file(dir_d):
        file_count = len(next(os.walk(dir_d))[2])
        return file_count


if __name__ == "__main__":
    ozka = Train_Former()
    ozka.accountant()

