import calendar
import time

from common.util.common_util import file_exists
from common.writer.FileWriteBase import FileWriteBase


class FileWriter(FileWriteBase):
    def __init__(self, file_name, extension):
        if file_exists(file_name, extension):
            ts = calendar.timegm(time.gmtime())
            # create copy based on the timestamp to ensure that previous one is not
            # overwrite with the new one.
            file_name = file_name.replace(".", f"_{ts}.")
        super(FileWriter, self).__init__(file_name, extension, "w+")
