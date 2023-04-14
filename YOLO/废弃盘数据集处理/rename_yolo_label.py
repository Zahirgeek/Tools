# 改变YOLO格式标注文件的类别

import os
import glob
import argparse

def process_label(label_path, class_id, target_id):
    """
    读取并处理YOLO的label文件
    :param label_path: label文件的位置, str
    :param class_id: 需要修改的id, int
    :param target_id: 修改后的id, int
    :return:
    """
    labels_path = os.path.join(label_path, "*.txt")
    labels_path_list = glob.glob(labels_path)
    assert len(labels_path_list)>0, "该路径下未读取到任何txt文件，请检查输入的路径是否正确"
    # 遍历labels_path_list读取txt文件
    for i, label in enumerate(labels_path_list):
        # print("正在读取{}".format(label))
        if os.path.getsize(label) == 0:
            # print('{}是空文件，跳过'.format(label))
            continue
        try:
            with open(label, "r", encoding='utf-8') as f:
                # 如果文件需要修改，该值为1
                change_label = 0
                new_txt_list = list()
                # txt文件中的内容
                txt_list = f.readlines()
                # 遍历txt中每一行数据
                for i, content in enumerate(txt_list):
                    # 用空格将每行的信息分割开
                    content_list = content.split(" ")
                    try:
                        # 将每行的第一个类别属性转为int
                        content_class = int(content_list[0])
                    except Exception as e:
                        print("int类型转换失败")
                        print(e)
                    if content_class == class_id:
                        # 替换类别
                        content_list[0] = str(target_id)
                        change_label = 1

                    new_content = " ".join(content_list)
                    new_txt_list.append(new_content)

            if change_label == 1:
                with open(label, "w", encoding='utf-8') as f:
                    f.writelines(new_txt_list)
                    print("{}文件修改完成".format(label))

        except Exception as e:
            print("{}读取失败".format(label))
            print(e)


def get_args():
    parser = argparse.ArgumentParser(description="改变YOLO格式标注文件的类别,使用方法:\n将标注类别为3的id改变为2\n"
                                                 "python rename_yolo_label.py -i {label_path} -c 3 -t 2")
    parser.add_argument("--input_path", "-i", type=str,
                        help="label文件的位置")
    parser.add_argument("--class_id", "-c", type=int,
                        help="需要修改的id号")
    parser.add_argument("--target_id", "-t", type=int,
                        help="修改后的id号")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # label文件位置
    args = get_args()
    label_path = args.input_path
    # label_path = os.path.join("task_1-2023_03_22_02_23_56-yolo 1.1", "obj_train_data")
    process_label(label_path, args.class_id, args.target_id)