import os


class FileWriteBase(object):
    def __init__(self, file_name, extension, mode):
        self.__mode = mode
        self.__file_name = file_name
        self.__extension = extension
        self.__temp_file_name = f"{file_name}{extension}.temp"
        self.__file = open(f"{file_name}{extension}", mode)
        self.__temp_file = None
        self.__contents = []

    def __sync(self):
        self.__file.flush()
        os.fsync(self.__file.fileno())

    def write_header(self, header):
        self.write(header)

    def write(self, content):
        if isinstance(content, list):
            content = str(content)[1:-1]
        if not content.endswith("\n"):
            content = content + "\n"
        self.__contents.append(content)
        self.__temp_file = open(self.__temp_file_name, self.__mode)
        self.__temp_file.writelines(self.__contents)
        self.__temp_file.close()
        self.__file.write(content)
        self.__sync()

    def close(self):
        self.__file.close()
        self.__contents.clear()
        if os.path.exists(self.__temp_file_name):
            os.remove(self.__temp_file_name)

    def plot(self, x_axis_key, y_axis_key, suffix=""):
        import pandas as pd
        import matplotlib.pyplot as plt
        df = pd.read_csv(f"{self.__file_name}{self.__extension}")
        df.plot(x=x_axis_key, y=y_axis_key)
        file_name = f"{self.__file_name}_{suffix}.png" if suffix != "" else f"{self.__file_name}.png"
        plt.savefig(file_name)
