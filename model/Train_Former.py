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
        self.save_list = [self.save_upper, self.save_lower, self.save_numbr]
        self.train_list = [self.train_upper, self.train_lower,
                      self.train_numbr, self.train_class]



    @property
    def Json_Self(self):
        with open("TrFo_Self.json") as f:
            file = json.load(f)
        Json_Self = file
        return Json_Self

    def accountant(self):
        Json_Self = {}
        Json_Self["Save_dir"] = {}
        Json_Self["Train_dir"] = {}
        for e in self.save_list:
            fcs_list = []
            dir_cnt, dir_ls = Train_Former.count_dir(e)
            dir_name_abr = re.search(r"([^/]+$)", str(e)).group(1)
            Json_Self["Save_dir"][dir_name_abr] = {}
            Json_Self["Save_dir"][dir_name_abr]["Dir_count"] = dir_cnt
            for i in dir_ls:
                file_cnt, file_list = Train_Former.count_file(e + "/" + i)
                fcs_list.append(file_cnt)
                min_value = min(fcs_list)
                total = sum(fcs_list)
                Json_Self["Save_dir"][dir_name_abr]["Min_fc"] = min_value
                Json_Self["Save_dir"][dir_name_abr]["Total_files"] = total
                Json_Self["Save_dir"][dir_name_abr][i] = file_cnt
        for e in self.train_list:
            fct_list = []
            dir_cnt, dir_ls = Train_Former.count_dir(e)
            dir_name_abr = re.search(r"([^/]+$)", str(e)).group(1)
            Json_Self["Train_dir"][dir_name_abr] = {}
            Json_Self["Train_dir"][dir_name_abr]["Dir_count"] = dir_cnt
            for i in dir_ls:
                file_cnt, file_list = Train_Former.count_file(e + "/" + i)
                fct_list.append(file_cnt)
                min_value = min(fct_list)
                total = sum(fct_list)
                Json_Self["Train_dir"][dir_name_abr]["Min_fc"] = min_value
                Json_Self["Train_dir"][dir_name_abr]["Total_files"] = total
                Json_Self["Train_dir"][dir_name_abr][i] = file_cnt
        with open("TrFo_Self.json", 'w') as f:
            json.dump(Json_Self, f, indent=4, sort_keys=True)

        return print(Json_Self)


    def File_Copy(self):
        self.accountant()
        self.Class_former()
        self.Classifajar_former()
        return print("Akapulko")


    def Class_former(self):

        sjson = self.Json_Self
        comp_dict_s = {}
        comp_dict_t = {}
        for key_s, value_s in sjson["Save_dir"].items():
            comp_dict_s[key_s] = value_s.get("Min_fc")
        for key_t, value_t in sjson["Train_dir"].items():
            comp_dict_t[key_t] = value_t.get("Min_fc")
        copy_dict = {}
        for key_s, value_s in comp_dict_s.items():
            for key_t, value_t in comp_dict_t.items():
                if key_s == key_t:
                    if value_s == None:
                        value_s = 0
                    elif value_t == None:
                        value_t = 0
                    copycount = value_s - value_t
                    copy_dict[key_s] = copycount
        for e in self.save_list:
            dir_name_abr_root = re.search(r"([^/]+$)", str(e)).group(1)
            dir_cnt, dir_ls = Train_Former.count_dir(e)
            for key, value in copy_dict.items():
                if dir_name_abr_root == key:
                    for i in dir_ls:
                        dir_name_abr_class = i
                        save_dir = e + "/" + i
                        target_dir = self.train_dir + "/" + dir_name_abr_root + "/"\
                                     + dir_name_abr_class
                        Min_fc = sjson["Save_dir"][dir_name_abr_root]["Min_fc"]
                        i_fcount, file_list_i = Train_Former.count_file(save_dir)
                        count_target, file_list = Train_Former.count_file(target_dir)
                        for c in range(count_target, Min_fc):
                            for file in file_list_i:
                                file_nr = re.search(r"([^_]+$)", str(file)).group(1)
                                file_nr = re.search(r"(.)", str(file_nr)).group(1)
                                if file_nr == str(c + 1):
                                    copied_file = Train_Former.read_file(save_dir, file)
                                    Train_Former.save_file(target_dir, file, copied_file)

        return print(copy_dict)


    def Classifajar_former(self):

        sjson = self.Json_Self
        comp_dict_cl = {}
        for key_s, value_s in sjson["Save_dir"].items():
            comp_dict_cl[key_s] = value_s.get("Total_files")
        classifajar_min = []
        for e in comp_dict_cl.items():
            classifajar_min.append(e[1])
        clasifaj_count = min(classifajar_min)
        for e in self.save_list:
            dir_name_abr_root = re.search(r"([^/]+$)", str(e)).group(1)
            dir_cnt, dir_ls = Train_Former.count_dir(e)
            target_dir = self.train_dir + "/" + "Classifajar" + "/" + dir_name_abr_root
            count_target, file_list = Train_Former.count_file(target_dir)
            for c, i in zip(range(count_target, clasifaj_count), dir_ls):
                save_dir = e + "/" + i
                i_fcount, file_list_i = Train_Former.count_file(save_dir)
                for file in file_list_i:
                    file_nr = re.search(r"([^_]+$)", str(file)).group(1)
                    file_nr = re.search(r"(.)", str(file_nr)).group(1)
                    nook = str(int(c // dir_cnt))
                    if eval(file_nr) == eval(nook) + 1:
                        copied_file = Train_Former.read_file(save_dir, file)
                        Train_Former.save_file(target_dir, file, copied_file)
                        break

        return print(comp_dict_cl)


    @staticmethod
    def read_file(dir_c, fname):
        if os.path.exists(dir_c) == False:
            os.makedirs(dir_c)
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
        if os.path.exists(dir_d) == False:
            os.makedirs(dir_d)
        dir_count = len(next(os.walk(dir_d))[1])
        dir_list = []
        for root, dirs, files in os.walk(dir_d):
            dir_list.append(dirs)

        return dir_count, dir_list[0]

    @staticmethod
    def count_file(dir_d):
        if os.path.exists(dir_d) == False:
            os.makedirs(dir_d)
        file_list = []
        file_count = len(next(os.walk(dir_d))[2])
        for root, dirs, files in os.walk(dir_d):
            file_list.append(files)

        return file_count, file_list[0]


if __name__ == "__main__":
    ozka = Train_Former()
    ozka.accountant()
    ozka.File_Copy()
    #ozka.Class_former()
