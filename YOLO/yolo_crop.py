# 将yolo标注的图片按标注区域裁剪出来

import cv2
import os
import glob
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="将yolo标注的图片按标注区域裁剪出来")
    parser.add_argument("--input_path", "-i", type=str,
                        help="图片文件和其对应label文件的位置")
    parser.add_argument("--output_path", "-o", type=str,
                        help="裁剪的图片输出的位置，目录为{output_path}/data/{类别}")
                        
    args = parser.parse_args()
    return args

def crop_image(origin_img,  x,  y,  w,  h):
    """
    裁剪图片
    origin_img:原图，ndarray
    x:yolo标注文件每行的第2个参数,框的中心点x坐标,float
    y:yolo标注文件每行的第3个参数,框的中心点y坐标,float
    w:yolo标注文件每行的第4个参数,框的宽,float
    h:yolo标注文件每行的第5个参数,框的高,float
    
    return:按照label文件抠出的图片, ndarray
    """
    img_width = origin_img.shape[1] 
    img_height = origin_img.shape[0]

    x,y,h,w = int(x*img_width), int(y*img_height), int(h*img_height), int(w*img_width)
    xmin = int(x - w/2)
    xmax =int( x + w/2)
    ymin = int(y - h/2)
    ymax = int(y + h/2)
    imgCrop = origin_img[ymin: ymax, xmin: xmax]

#     cv2.imshow("crop image", imgCrop)
#     cv2.waitKey(0)
    
    return imgCrop
    
def create_folders(path):
    """
    如果path不存在就创建
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print("{}目录不存在，已被创建".format(path))
        
def image_process(input_path, output_path):
    """
    读取指定文件夹下的.jpg .png图片文件，并进行抠图
    input_path:图片和其对应label文件位置
    """
    jpg_path = os.path.join(input_path, "*.jpg")
    jpg_list = glob.glob(jpg_path)
    png_path = os.path.join(input_path, "*.png")
    png_list = glob.glob(png_path)
    
    image_path_list = list()
    image_path_list.extend(jpg_list)
    image_path_list.extend(png_list)
    
    print("需要裁剪的图片数量：{}".format(len(image_path_list)))
    # 计算裁剪类别对应的图片数量
    count_dict = dict()
        
    for i, img_path in enumerate(image_path_list):
        img_basepath = os.path.splitext(img_path)[0]
        # 图片对应的label name
        label_path = img_basepath+".txt"
        if not os.path.exists(label_path):
            print("{}对应的label文件不存在，跳过该图片".format(os.path.basename(img_path)))
            continue
        
        # 读取label文件的内容
        content = list()
        with open(label_path,encoding='utf-8') as file:
            content=file.readlines()
        if not content:
            continue
            
        # 读取图片
        try:
            origin_img = cv2.imread(img_path)
        except Exception as e:
            print(e)
            continue
        
        img_basename = os.path.basename(img_path)
        
        # 遍历content，获取每张图的标注信息
        for i, data in enumerate(content):
            data_list = data.split()
            if len(data_list) != 5:
                continue
            index, xc, yc, w, h = data_list
            
            xc = float(xc)
            yc = float(yc)
            w = float(w)
            h = float(h)
            imgCrop = crop_image(origin_img,  xc,  yc,  w,  h)
            
            # TODO:按照index分类保存抠图
            # 创建文件夹
            img_folder_name = "img"+index
            output_path_ = os.path.join(output_path, "data", img_folder_name)
            create_folders(output_path_)
            img_name, img_suffix = os.path.splitext(img_basename)
            img_basename_new = img_name + str(i) + img_suffix
            save_path = os.path.join(output_path_, img_basename_new)

            # 保存裁剪的图片
            try:
                cv2.imwrite(save_path, imgCrop)
            except Exception as e:
                print("{}保存失败".format(save_path))
                print(e)
            else:
                # 计数
                if index not in count_dict:
                    count_dict[index] = 1
                else:
                    count_dict[index] += 1
                
    for key, value in count_dict.items():
        print("类别{}被裁剪出的数量：{}".format(key, value))
        
if __name__ == '__main__':
    args = get_args()
    input_path = args.input_path
    output_path = args.output_path
    image_process(input_path, output_path)