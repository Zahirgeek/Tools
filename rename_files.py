# 重命名文件名，防止复制文件产生冲突
import os
import argparse


def rename_files(args):
    '''
    在指定的文件夹内按用户输入的规则重命名文件名
    :param args:
    :return:
    '''

    file_path = args.file_path
    assert os.path.isdir(file_path), "路径输入应为文件夹"
    oldname_list = os.listdir(file_path)
    newname_prefix = args.file_name
    newname_num = args.number

    for i, oldname in enumerate(oldname_list):
        old_path = os.path.join(file_path, oldname)
        if not os.path.isfile(old_path):
            continue
        _, suffix = os.path.splitext(oldname)
        newname = "{}_{}{}".format(newname_prefix, newname_num, suffix)
        new_path = os.path.join(file_path, newname)
        try:
            # 重命名文件
            os.rename(old_path, new_path)
        except Exception as e:
            print(e)
            print("-"*10)
        # 文件名数字自加
        newname_num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="重命名文件名，防止复制文件产生冲突")
    parser.add_argument("--file_path", "-f", type=str,
                        help="文件所在位置")
    parser.add_argument("--file_name", "-n", type=str,
                        help="文件名称，新文件命名规则：文件名称_数字")
    parser.add_argument("--number", "-num", type=int, default=0,
                        help="文件名包含的数字，默认从0开始")

    args = parser.parse_args()
    rename_files(args)