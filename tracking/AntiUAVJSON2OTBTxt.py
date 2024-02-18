import json
import os
import glob

# 设置文件夹路径
folder_path = '.'  # 替换为你的文件夹路径

# 遍历文件夹下的子文件夹
for root, dirs, files in os.walk(folder_path):
    for dir_name in dirs:
        # 获取子文件夹路径
        subfolder_path = os.path.join(root, dir_name)

        # 查找子文件夹中的JSON文件
        json_files = glob.glob(os.path.join(subfolder_path, '*.json'))

        # 循环处理每个JSON文件
        for json_file in json_files:
            # 读取JSON文件
            with open(json_file, 'r') as f:
                data = json.load(f)

            # 获取文件路径和名称
            file_dir = os.path.dirname(json_file)
            file_name = os.path.splitext(os.path.basename(json_file))[0]

            # 创建txt文件
            txt_file = os.path.join(file_dir, file_name + '.txt')
            with open(txt_file, 'w') as f:
                # 遍历每一帧的数据
                for exist, rect in zip(data['exist'], data['gt_rect']):
                    if exist == 1:
                        # 写入矩形的四个角点坐标
                        line = ','.join(str(coord) for coord in rect)
                    else:
                        # 如果exist为0，写入NaN
                        line = 'NaN,NaN,NaN,NaN'
                    f.write(line + '\n')

            print('转换完成！生成的文本文件为:', txt_file)