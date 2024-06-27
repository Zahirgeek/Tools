# 将yolo标注的图片按标注区域裁剪出来，并按自定义尺寸填充像素

import cv2
import os
import glob
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="将yolo标注的图片按标注区域裁剪出来，并按自定义尺寸填充像素")
    parser.add_argument("--input_path", "-i", type=str,
                        help="图片文件和其对应label文件的位置")
    parser.add_argument("--output_path", "-o", type=str,
                        help="裁剪的图片输出的位置，目录为{output_path}/{类别}")
    parser.add_argument("--image_size", "-is", type=int,
                        help="经过填充后的图片大小")
                        
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
    xmax = int(x + w/2)
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

def image_padding(img, image_size):
    """
    在裁切后的图片上补像素
    img: 图片，ndarray
    image_size: padding后的大小, int
    return: padding后的图片
    """
    top_size = int((image_size - img.shape[0]) / 2)
    bottom_size = int((image_size - img.shape[0]) / 2)
    left_size = int((image_size - img.shape[1]) / 2)
    right_size = int((image_size - img.shape[1]) / 2)
    if top_size<0 or bottom_size<0 or left_size<0 or right_size<0:
        print("填充失败，用户设置的填充图片后的大小:{} 比填充前的图片大小:({}, {})要小，请重新设置image_size".format(image_size, img.shape[1], img.shape[0]))
        return
#     print("top_size: {}, bottom_size: {}, left_size: {}, right_size: {}".format(top_size, bottom_size, left_size, right_size))
    process_img = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)
    
    return process_img
    
def image_process(input_path, output_path, image_size):
    """
    读取指定文件夹下的.jpg .png图片文件，并进行抠图、填充像素，保存填充后的图片和对应的label
    input_path:图片和其对应label文件位置, string
    output_path:输出裁剪后填充像素的图片和对应label文件位置, string
    image_size:padding后的大小,int
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
            
            # 创建文件夹
            img_folder_name = "img"+index
            output_path_ = os.path.join(output_path, img_folder_name)
            create_folders(output_path_)
            
            img_name, img_suffix = os.path.splitext(img_basename)
            img_basename_new = img_name + str(i)
            save_path = os.path.join(output_path_, img_basename_new + img_suffix)
            
            try:
                padding_img = image_padding(imgCrop, image_size)
            except Exception as e:
                print("{}图片padding失败".format(img_path))
                print(e)
                continue
                
            if padding_img is None:
                continue
                    
            # 保存填充的图片
            try:
                cv2.imwrite(save_path, padding_img)
            except Exception as e:
                print("{}保存失败".format(save_path))
                print(e)
                continue
            else:
                # 计数
                if index not in count_dict:
                    count_dict[index] = 1
                else:
                    count_dict[index] += 1
            
            # 保存填充图片对应的label文件
            label_path = os.path.join(output_path_, img_basename_new + ".txt")
            with open(label_path, "w") as f:
                f.write(index + ' ' + '0.500000' + ' ' + '0.500000' + ' ' + str(format(imgCrop.shape[1] / image_size, '.6f')) + ' ' + str(format(imgCrop.shape[0] / image_size, '.6f')))
            
                
    for key, value in count_dict.items():        
        print("类别{}被裁剪并填充出的数量：{}".format(key, value))
    print("图片裁剪和填充已完成！")
        
if __name__ == '__main__':
    args = get_args()
    input_path = args.input_path
    output_path = args.output_path
    image_size = args.image_size
    image_process(input_path, output_path, image_size)