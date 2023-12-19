import os
import shutil


path, _num = "./", 0
for i in os.listdir(path):
    src_path = os.path.join(path, i)
    if not os.path.isdir(src_path):
        continue
    file_list = os.listdir(src_path)
    file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))
    for j in os.listdir(src_path):
        file = os.path.join(src_path, j)
        new_name = os.path.join(src_path, "{}.png".format(_num))
        os.rename(file, new_name)
        # new_file = path + "{}.png".format(_num)
        # shutil.move(new_name, new_file)
        _num += 1



