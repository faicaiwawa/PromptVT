import os
import fnmatch
'''
Place this file in the trace results directory (directory contains *model_time and *time)
'''

filePath= '.'
file_number = 0
sum_all_file = 0
for f_name in os.listdir(filePath):
    if fnmatch.fnmatch(f_name, '*model_time.txt'):
        file_number = file_number + 1
        with open(f_name, "r") as f:
            sum_one_file = 0
            line_number = 0
            for line in f:
                wordlist = line.split()
                for a in wordlist:
                    line_number=line_number + 1
                    time = float(a)
                    sum_one_file = sum_one_file + time
        sum_all_file = sum_all_file + line_number/sum_one_file
        print('{}:{}'.format(f_name,line_number/sum_one_file))
        f.close()
print(' ')
print('average_fps : {}'.format(sum_all_file/file_number))
print(' ')
print('sum_all_file_fps : {}'.format(sum_all_file))
print(' ')
print('file_number : {}'.format(file_number))

