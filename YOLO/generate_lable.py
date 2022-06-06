# 为图片生成空label
import os

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def generate_lables(label_path, images_path):
    '''
    为对应图片生成同名的label文件
    label_path: 生成label文件的目录
    images_path: 图片路径
    '''

    images_list = os.listdir(images_path)
    create_folder(label_path)
    for i, img in enumerate(images_list):
        image_name = img.split('.')[:-1]
        if isinstance(image_name, list):
            image_name = ".".join(image_name)
        lable_name = '{}.txt'.format(image_name)
        
        with open(os.path.join(label_path, lable_name), 'w') as f_out:
            pass

if __name__ == '__main__':
    # 在指定文件夹生成对应图片label
    generate_lables('label', 'img')