import os
import pathlib
import re
import time
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import scipy as sp
import scipy.stats as sp_stats
import pickle as pkl

DATA_DIRECTORY_RAW_FILE_TXT = "../ML_Trajectories_Raw"
DATA_DIRECTORY_RAW_FILE_CSV = "../ML_Trajectories"
DATA_DIRECTORY_TIME_SERIES = "ML_TimeSeries"
DATA_DIRECTORY_MODEL_WEIGHTS = "ML_ModelWeights"

OUTPUT_DIRECTORY_GIF = "GIF_Output"

DATAFRAME_COLUMN_TIME_SERIES = ["index", "LTS", "SST", "Subsidence", "Night_Day", "RH", "q", "wsp", "TCC", "UnobstructedLC"]
DATAFRAME_COLUMN_TIME_SERIES_INPUT_LABEL = DATAFRAME_COLUMN_TIME_SERIES[1:-2]
DATAFRAME_COLUMN_TIME_SERIES_OUTPUT_LABEL = DATAFRAME_COLUMN_TIME_SERIES[-2:]

DATAFRAME_COLUMN_TIME_SERIES_ANORMALY = ["index", "LTS", "SST", "Subsidence", "Night_Day", "RH", "q", "wsp", "LTS_A", "SST_A", "Subsidence_A", "RH_A", "q_A", "wsp_A", "TCC"]
DATAFRAME_COLUMN_TIME_SERIES_INPUT_LABEL_ANORMALY = DATAFRAME_COLUMN_TIME_SERIES[1:-1]
DATAFRAME_COLUMN_TIME_SERIES_OUTPUT_LABEL_ANORMALY = DATAFRAME_COLUMN_TIME_SERIES[len(DATAFRAME_COLUMN_TIME_SERIES)-1:len(DATAFRAME_COLUMN_TIME_SERIES)]

DATAFRAME_COLUMN_TIME_SERIES_EXTEND = ["index", "LTS", "SST", "Subsidence", "Night_Day", "RH", "q", "wsp", "lat", "lon", "day", "year", "TCC", "UnobstructedLC"]
DATAFRAME_COLUMN_TIME_SERIES_INPUT_LABEL_EXTEND = DATAFRAME_COLUMN_TIME_SERIES[1:-2]
DATAFRAME_COLUMN_TIME_SERIES_OUTPUT_LABEL_EXTEND = DATAFRAME_COLUMN_TIME_SERIES[-2:]

DATAFRAME_COLUMN_TIME_SERIES_ERA5 = ["index", "LTS", "SST", "Subsidence", "Night_Day", "RH", "q", "wsp", "TCC", "ERA5"]
DATAFRAME_COLUMN_TIME_SERIES_INPUT_LABEL_ERA5 = DATAFRAME_COLUMN_TIME_SERIES_ERA5[1:-2]
DATAFRAME_COLUMN_TIME_SERIES_OUTPUT_LABEL_ERA5 = DATAFRAME_COLUMN_TIME_SERIES_ERA5[-2:]

def create_directory(dir_path):
    """
    Create directory. 
    
    This function can only create the directory one level at a time. 
    i.e. To create level1/level2/level3, level1/level2 must exists.
    """
    if (not os.path.isdir(dir_path)):
        os.mkdir(dir_path)
    return

def filter_file_names(file_name_pattern, file_directory=None):
    """
    return all filename that matches the file pattern. 
    
    file_name_pattern: string, Regular Expression
    file_directory:    string, The file directory to search. If not specified, then current directory will be searched. 
    return:            A list of file that matches the file_name_pattern
    """
    if (file_directory is None):
        file_directory = "./"
    RE = re.compile(file_name_pattern)
    ret_val = []
    for file_name in os.listdir(file_directory):
        if (not RE.match(file_name) is None):
            ret_val.append(file_name)
    return ret_val

def split_list(a_list, ratio):
    """
    split a list into several list. The number of return list is specify by len(ratio). 
    The ratio will have +- 1 difference
    return type will be python list (not np.array)
    
    a_list: list
    ratio:  list-like, each element specifies the ratio of size for each output list. The ratio will be normalized
    return: [list1, list2, ...]
    """
    
    def normalize_ratio(a_len, ratio):
        ratio = np.array(ratio)
        ratio_out = np.floor(ratio / np.sum(ratio) * a_len)
        plus_1_list = np.random.choice(np.arange(ratio.shape[0]), int(a_len - np.sum(ratio_out)), replace=False)
        for idx in plus_1_list:
            ratio_out[idx] += 1
        return ratio_out
    
    def random_split_into_2_list(a_list, output_1_size):
        """
        this will randomly split a_list into two lists, where the size of the list1 is output_1_size

        return (list1, list2)
        """
        list1_sample_idx = np.random.choice(np.arange(len(a_list)), int(output_1_size), replace=False)

        list1 = []
        list2 = []
        for idx in np.arange(len(a_list)):
            if (idx in list1_sample_idx):
                list1.append(a_list[idx])
            else:
                list2.append(a_list[idx])

        return (list1, list2)
    
    ratio_lens = normalize_ratio(len(a_list), ratio)
    
    ret_val = [] # output list
    temp_list = a_list[:] # make a copy of original list
    for idx in np.arange(len(ratio)):
        list1, list2 = random_split_into_2_list(temp_list, ratio_lens[idx])
        ret_val.append(list1)
        temp_list = list2
        
    return ret_val

def get_runtime_marker():
    """
    Get the marker indicating the starting time
    """
    return time.perf_counter_ns()
    
def get_runtime_in_nanosecond(runtime_marker):
    """
    Get the runtime since the runtime_marker in nanosecond
    """
    return (time.perf_counter_ns() - runtime_marker)

def get_runtime_in_second(runtime_marker):
    """
    Get the runtime since the runtime_marker in second
    Use this function in conjunction with format_time_s_2_hms() to print in 
    more readable format
    """
    return get_runtime_in_nanosecond(runtime_marker) / 1e9

def format_time_s_2_hms(second):
    """
    format time (input in seconds as float or int)
        eg. 31215 -> "8h 40m 15s"
    """
    second = int(second)
    if (second > 3600):  
        return "{0}h {1:>2}m {2:>2}s".format(second // 3600, second % 3600 // 60, second % 60)
    elif (second > 60):
        return "{0}m {1:>2}s".format(second // 60, second % 60)
    else: 
        return "{0}s".format(second)

def return_all_filename_at_directory(directory):
    """
    return all filename under given directory as list of strings
    """
    if (directory == None): 
        directory == "."
    return os.popen("ls {0}".format(directory)).read().split("\n")

def print_all_filename_at_directory(directory):
    """
    print all filename under given directory
    """
    if (directory == None): 
        directory == "."
    print(os.popen("ls {0}".format(directory)).read())
    
    

def create_empty_time_series_data():
    """
    Create an array of empty time-series data
    """
    df_time_series = []
    for idx in np.arange(9):
        df_time_series.append(pd.DataFrame(columns=DATAFRAME_COLUMN_TIME_SERIES))
    return df_time_series[:]  # return a copy

def create_empty_time_series_data_anormaly():
    """
    Create an array of empty time-series data
    """
    df_time_series = []
    for idx in np.arange(9):
        df_time_series.append(pd.DataFrame(columns=DATAFRAME_COLUMN_TIME_SERIES_ANORMALY))
    return df_time_series[:]  # return a copy

def create_empty_time_series_data_extended():
    """
    Create an array of empty time-series data
    """
    df_time_series = []
    for idx in np.arange(9):
        df_time_series.append(pd.DataFrame(columns=DATAFRAME_COLUMN_TIME_SERIES_EXTEND))
    return df_time_series[:]  # return a copy

def create_empty_time_series_data_ERA5():
    """
    Create an array of empty time-series data
    """
    df_time_series = []
    for idx in np.arange(9):
        df_time_series.append(pd.DataFrame(columns=DATAFRAME_COLUMN_TIME_SERIES_ERA5))
    return df_time_series[:]  # return a copy
    
def read_time_series_data(folder_name):
    """
    read time series data from folder
    """
    df_time_series = []
    for idx in np.arange(9):
        df_time_series.append(
            pd.read_csv(
                "{0}/{1}/T{2}.csv".format(DATA_DIRECTORY_TIME_SERIES, folder_name, idx), 
            )
        )
    return df_time_series[:]  # return a copy

def save_tensors(tensors, folder_name):
    """
    save the tensor into the given folder
    
    tensors: array-liked tensors (regular python list) or a single tensor
    folder_name: the string representing the name of the folder
    """
    
    if (type(tensors) is not list):
        tensors = [tensors]
    
    create_directory("{0}/{1}".format(DATA_DIRECTORY_MODEL_WEIGHTS, folder_name))
    
    for idx in range(len(tensors)):
        torch.save(tensors[idx], "{0}/{1}/MW{2}.tensor".format(DATA_DIRECTORY_MODEL_WEIGHTS, folder_name, idx))

def read_tensors(folder_name):
    """
    read tensors for given folder. This is the inverse of the save_tensor() when tensors is a list.
    
    folder_name: the string representing the name of the folder
    require_grad: set default require_grad for loading 
    
    return tensors: array-liked tensors (regular python list)
    """
    
    # find and filter out all model weights
    file_names = os.listdir("{0}/{1}".format(DATA_DIRECTORY_MODEL_WEIGHTS, "test"))
    tensor_file_names = [file_name for file_name in file_names if ((len(file_name) >= 7) and (file_name[-7:] == ".tensor") and (file_name[:2] == "MW"))]
     
    ret_list = []
    for idx in range(len(tensor_file_names)):
        ret_list.append(torch.load("{0}/{1}/MW{2}.tensor".format(DATA_DIRECTORY_MODEL_WEIGHTS, folder_name, idx)))
    return ret_list[:]  # return a copy

def save_time_series_data(folder_name, df_time_series):
    """
    save file time series data to folder
    """
    create_directory("{0}/{1}".format(DATA_DIRECTORY_TIME_SERIES, folder_name))
    
    for idx in np.arange(9):
        df_time_series[idx].to_csv(
            "{0}/{1}/T{2}.csv".format(DATA_DIRECTORY_TIME_SERIES, folder_name, idx), 
            index=False
        )
    return 

def powerset(fullset, keep_shorter_than=None, keep_equal_to=None, keep_longer_than=None):
    """
    Return the powerset of the given full set. The list will be sorted by length of elements
    If keep_* parameters are specified, the returned powerset will be filtered 
        based on the condition
    """
    
    listrep = list(fullset)
    n = len(listrep)
    full_powerset = [[listrep[k] for k in range(n) if i&1<<k] for i in range(2**n)]
    if (keep_shorter_than is None and keep_equal_to is None and keep_longer_than is None):
        full_powerset.sort(key=lambda s: len(s))
        return full_powerset
    else:
        ret_val = []
        if (keep_shorter_than is not None):
            ret_val += [element for element in full_powerset if (len(element) < keep_shorter_than)]
        if (keep_equal_to is not None):
            ret_val += [element for element in full_powerset if (len(element) == keep_equal_to)]
        if (keep_longer_than is not None):
            ret_val += [element for element in full_powerset if (len(element) > keep_longer_than)]
        ret_val.sort(key=lambda s: len(s))
        return ret_val
    
def set_difference(A, B):
    """
    This is a list operation which will preserve the order of the original element
    Return the result A - B
    """
    return [a for a in A if (a not in B)]

def softclamp(x, low, high, low_slope, high_slope):
    """
    soft_clamp function
    Any input outside range [low, high] will multiple the corresponding slope
    """
    y = x
    y[x < low] = (x[x < low] - low) * low_slope + low
    y[x > high] = (x[x > high] - high) * high_slope + high
    return y


class Normalizer:
    """
    The class that will handle the normalization and denormalization of the 
    input and output data
    """
    def __init__(self):
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        
    def normalize_input(self, input_data):
        """
        input_data:  torch.tensor
        """
        if ((self.input_mean is None) or (self.input_std is None)):
            print("mean and std are not set")
            return input_data
        return (input_data - self.input_mean) / self.input_std
    
    def denormalize_input(self, input_data):
        """
        input_data:  torch.tensor
        """
        if ((self.input_mean is None) or (self.input_std is None)):
            print("mean and std are not set")
            return input_data
        return input_data * self.input_std + self.input_mean
    
    def normalize_output(self, output_data):
        """
        output_data: torch.tensor
        """
        if ((self.output_mean is None) or (self.output_mean is None)):
            print("mean and std are not set")
            return output_data
        return (output_data - self.output_mean) / self.output_std
    
    def denormalize_output(self, output_data):
        """
        output_data: torch.tensor
        """
        if ((self.output_mean is None) or (self.output_mean is None)):
            print("mean and std are not set")
            return output_data
        return output_data * self.output_std + self.output_mean
    
    def set_mean_and_sd(self, input_data, output_data):
        """
        input_data:  torch.tensor
        output_data: torch.tensor
        """

        # input data
        input_mean_temp = []
        input_std_temp = []
        for idx in np.arange(input_data.shape[1]):
            temp_arr = input_data[:, idx]
            mean = torch.mean(temp_arr)
            std = torch.std(temp_arr)
            
            # solve devided by zero issue caused by std = 0
            if (torch.abs(std) < 1e-7):
                std = 1  # do not devided by std
                
            input_mean_temp.append(mean)
            input_std_temp.append(std)
            
        self.input_mean = torch.tensor(input_mean_temp)
        self.input_std = torch.tensor(input_std_temp)
                
        # output data
        output_mean_temp = []
        output_std_temp = []
        for idx in np.arange(output_data.shape[1]):
            temp_arr = output_data[:, idx]
            mean = torch.mean(temp_arr)
            std = torch.std(temp_arr)
            
            # solve devided by zero issue caused by std = 0
            if (torch.abs(std) < 1e-7):
                std = 1  # do not devided by std
                
            output_mean_temp.append(mean)
            output_std_temp.append(std)
            
        self.output_mean = torch.tensor(output_mean_temp)
        self.output_std = torch.tensor(output_std_temp)
        return self
        
    def normalize_input_nparray(self, input_data):
        """
        input_data:  np.array
        """
        return self.normalize_input(torch.tensor(input_data)).detach().numpy()
    
    def denormalize_input_nparray(self, input_data):
        """
        input_data:  np.array
        """
        return self.denormalize_input(torch.tensor(input_data)).detach().numpy()
    
    def normalize_output_nparray(self, output_data):
        """
        output_data: np.array
        """
        return self.normalize_output(torch.tensor(input_data)).detach().numpy()
    
    def denormalize_output_nparray(self, output_data):
        """
        output_data: np.array
        """
        return self.denormalize_output(torch.tensor(input_data)).detach().numpy()
        
    def set_mean_and_sd_nparray(self, input_data, output_data):
        """
        input_data:  torch.tensor
        output_data: torch.tensor
        """
        set_mean_and_sd(
            self, 
            torch.tensor(input_data), 
            torch.tensor(output_data)
        )
               
class ProgressBar:
    def __init__(self, total, val=None):
        self.total = total
        if (val is None):
            self.val = 0
        else:
            self.val = val
        self.msg = ""
        self.length = 40
        
        self.__last_output_string_length = 0
        
        self.__is_ended = False
        
    def update(self, val=None, msg=None):
        if (self.__is_ended):
            return
        if (val is not None):
            self.update_val(val)
        if (msg is not None):
            self.update_msg(msg)
           
        # clear last result
        self.clear_line()
        
        out_str = ""
        # percentage
        out_str += "[{0}/{1}|{2:.1f}%] ".format(self.val, self.total, self.get_percent_0_100())
        
        # bar
        tick = int(np.round(self.get_percent_0_1() * self.length, 0))
        blank = int(self.length - tick)
        out_str += "[{0}{1}]".format("="*tick, " "*blank)
        
        # msg
        if (self.msg is not None):
            out_str += " {0}".format(self.msg)
            
        # print
        print(out_str, end="\r")
        self.__last_output_string_length = len(out_str)
        return self
        
    def update_val(self, val):
        if (0 <= val and val <= self.total):
            self.val = val
        return self
        
    def update_msg(self, msg):
        self.msg = msg
        return self
    
    def clear_line(self):
        print(" " * self.__last_output_string_length, end="\r")
        
    def get_percent_0_100(self):
        return float(self.val * 100) / self.total
    
    def get_percent_0_1(self):
        return float(self.val) / self.total
    
    def set_total(self, total):
        self.total = total
        return self
    
    def inc_val(self, a=None):
        if (a is None):
            self.val += 1
        else:
            self.val += a
        return self
    
    def end(self):
        if (not self.__is_ended):
            self.__is_ended = True
            print()
        return self
    
    def generator(arr, msg=None):
        P = ProgressBar(len(arr))
        if (msg is not None):
            P.update_msg(msg)
        P.update()
        for i in arr:
            P.inc_val().update()
            yield i
        P.end().update()
        
class TimedProgressBar(ProgressBar):
    def __init__(self, total, val=None):
        super().__init__(total, val)
        self.__time_mark = get_runtime_marker()
        self.msg2 = ""
        
    def reset_timer(self):
        self.__time_mark = get_runtime_marker()
        return self
        
    def update(self, val=None, msg=None):
        if (val is not None):
            self.update_val(val)
        if (msg is not None):
            self.update_msg(msg)
        T_running = self.__time_mark
        
        if (self.val == 0): 
            self.msg = self.msg2 
        else:
            self.msg = "[{0} / {1}] {2}".format(
                format_time_s_2_hms(get_runtime_in_second(T_running)),
                format_time_s_2_hms(get_runtime_in_second(T_running) / self.get_percent_0_1()),
                self.msg2
            )
        super().update()
        return self
        
    def update_msg(self, msg):
        self.msg2 = msg
        return self
    
    def generator(arr, msg=None):
        P = TimedProgressBar(len(arr))
        if (msg is not None):
            P.update_msg(msg)
        P.update()
        for i in arr:
            P.inc_val().update()
            yield i
        P.end().update()
    
def get_current_day_time_string():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

class Bias_Filter:
    def __init__(self):
        self.m = None
        self.b = None
        self.cap = None
        self.__default_foward_with_cap = False
        
    def __call__(self, a):
        if (self.__default_foward_with_cap):
            return self.forward_with_cap(a)
        else:
            return self.forward(a)
        
    def forward(self, a):
        if ((self.b is None) or (self.m is None)):
            print("[Warning] Bias Filter hasn't trained yet")
            return None
        return (a - self.b) / self.m
    
    def forward_with_cap(self, a, cap=None):
        """
        cap must has format: (low, high)
        """
        if (cap is None): 
            cap = self.cap
            
        if (cap is None):
            print("[Warning] Bias Filter doesn't have cap")
            
        return np.minimum(np.maximum(self.forward(a), cap[0]), cap[1])
        
    
    def train(self, a_true, a_fit):
        result = sp_stats.linregress(a_true, a_fit)
        
        self.m = result.slope
        # if the slope is too shallow (very close to 0), then ignore it.
        if (np.abs(self.m) < 1e-6):
            self.m = 1
            
        self.b = result.intercept
        
        return self
    
    def set_cap(self, low, high):
        self.cap = (low, high)
        return self
    
    def default_with_cap(self):
        self.__default_foward_with_cap = True
        return self
    
    def default_without_cap(self):
        self.__default_foward_with_cap = False
        return self
    
class Archive:
    def __init__(self, file_path):
        self.file_path = pathlib.Path(file_path)

    def save_data(self, data, file_name):
        with open(self.file_path / file_name, 'wb') as f:
            pkl.dump(data, f)

    def load_data(self, file_name):
        with open(self.file_path / file_name, 'rb') as f:
            return pkl.load(f)