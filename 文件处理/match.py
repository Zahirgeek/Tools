'''
用第一个文件夹内容匹配第二个文件夹的内容，将匹配到的第一个文件夹中的文件输出
'''
import argparse
import os
import fnmatch
from shutil import move, copy


def get_args():
    parser = argparse.ArgumentParser(description="用第一个文件夹内容匹配第二个文件夹的内容，将匹配到的第一个文件夹中的文件输出")
    parser.add_argument("--first_folder", '-f1', type=str,
                        help="第一个文件夹位置")
    parser.add_argument("--second_folder", '-f2', type=str,
                        help="第二个文件夹位置")
    parser.add_argument("--output_path", '-o', type=str, default="",
                        help="输出的文件夹，如不指定，则默认为该程序所在的文件夹下的cp或mv文件夹内")
    parser.add_argument("--type", type=str, default="c",
                        help="Copy or Move?, input c or copy for copy / m or move for move, default is copy.")

    args = parser.parse_args()
    return args


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def copy_move(file_path, args):
    if not args.output_path:
        if args.type == 'c' or args.type == 'copy':
            to_path = os.path.join('cp')
        elif args.type == 'm' or args.type == 'move':
            to_path = os.path.join('mv')
    else:
        to_path = os.path.join(args.output_path)

    mkdirs(to_path)

    if args.type == 'c' or args.type == 'copy':
        copy(file_path, to_path)
    elif args.type == 'm' or args.type == 'move':
        print('-'*10, file_path)
        move(file_path, to_path)


def match_files(args):
    first_folder_list: list = os.listdir(args.first_folder)
    second_folder_list: list = os.listdir(args.second_folder)
    assert first_folder_list != [], "第一个文件夹是空的"
    assert second_folder_list != [], "第二个文件夹是空的"

    first_folder_set: set = set(first_folder_list)
    second_folder_set: set = set(second_folder_list)

    for i, first_file in enumerate(first_folder_set):
        # 文件名
        file_name = first_file.split('.')[0]
        for j, second_file in enumerate(second_folder_set):
            if fnmatch.fnmatch(second_file, file_name + '.*'):
                first_file_path = os.path.join(args.first_folder, first_file)
                try:
                    copy_move(first_file_path, args)
                except Exception as e:
                    print(e)

        progress_bar = i/(len(first_folder_set)-1)*100
        print("\r{:.1f}%".format(progress_bar), end="")


if __name__ == '__main__':
    args = get_args()
    match_files(args)
    print('\nfinished')
