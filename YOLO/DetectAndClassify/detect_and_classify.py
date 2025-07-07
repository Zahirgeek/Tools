#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 目标检测和分类脚本
功能：对图像和视频进行目标检测，并根据检测结果进行分类保存
"""

import os
import argparse
import shutil
from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import logging
from typing import List, Tuple, Dict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLODetectionClassifier:
    """YOLO检测分类器"""
    
    def __init__(self, model_path: str, classes_path: str, confidence_threshold: float = 0.5):
        """
        初始化检测分类器
        
        Args:
            model_path: YOLO模型路径
            classes_path: 类别文件路径
            confidence_threshold: 置信度阈值
        """
        self.model_path = model_path
        self.classes_path = classes_path
        self.confidence_threshold = confidence_threshold
        
        # 加载模型
        logger.info(f"正在加载YOLO模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 加载类别
        self.classes = self._load_classes(classes_path)
        logger.info(f"加载了 {len(self.classes)} 个类别: {self.classes}")
    
    def _load_classes(self, classes_path: str) -> List[str]:
        """加载类别文件"""
        with open(classes_path, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return classes
    
    def detect_image(self, image_path: str) -> Tuple[List[str], np.ndarray, dict]:
        """
        检测单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            detected_classes: 检测到的类别列表
            image: 原图像
            results_info: 检测结果信息
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return [], None, {}
        
        # 进行检测
        results = self.model(image_path, conf=self.confidence_threshold)
        
        detected_classes = []
        results_info = {
            'detections': [],
            'class_counts': {}
        }
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls.cpu().numpy()[0])
                    confidence = float(box.conf.cpu().numpy()[0])
                    
                    if cls_id < len(self.classes):
                        class_name = self.classes[cls_id]
                        if class_name not in detected_classes:
                            detected_classes.append(class_name)
                        
                        # 记录检测信息
                        bbox = box.xyxy.cpu().numpy()[0].tolist()
                        results_info['detections'].append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': bbox
                        })
                        
                        # 统计类别数量
                        if class_name not in results_info['class_counts']:
                            results_info['class_counts'][class_name] = 0
                        results_info['class_counts'][class_name] += 1
        
        return detected_classes, image, results_info
    
    def detect_video_frame(self, frame: np.ndarray) -> Tuple[List[str], dict]:
        """
        检测视频帧
        
        Args:
            frame: 视频帧
            
        Returns:
            detected_classes: 检测到的类别列表
            results_info: 检测结果信息
        """
        results = self.model(frame, conf=self.confidence_threshold)
        
        detected_classes = []
        results_info = {
            'detections': [],
            'class_counts': {}
        }
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls.cpu().numpy()[0])
                    confidence = float(box.conf.cpu().numpy()[0])
                    
                    if cls_id < len(self.classes):
                        class_name = self.classes[cls_id]
                        if class_name not in detected_classes:
                            detected_classes.append(class_name)
                        
                        bbox = box.xyxy.cpu().numpy()[0].tolist()
                        results_info['detections'].append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': bbox
                        })
                        
                        if class_name not in results_info['class_counts']:
                            results_info['class_counts'][class_name] = 0
                        results_info['class_counts'][class_name] += 1
        
        return detected_classes, results_info
    
    def visualize_detection(self, image: np.ndarray, results_info: dict) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原图像
            results_info: 检测结果信息
            
        Returns:
            visualized_image: 可视化后的图像
        """
        vis_image = image.copy()
        
        for detection in results_info['detections']:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return vis_image

class FileProcessor:
    """文件处理器"""
    
    def __init__(self, input_path: str, output_path: str, detector: YOLODetectionClassifier, 
                 visualize: bool = False):
        """
        初始化文件处理器
        
        Args:
            input_path: 输入路径
            output_path: 输出路径
            detector: YOLO检测器
            visualize: 是否可视化
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.detector = detector
        self.visualize = visualize
        
        # 支持的图像格式
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        # 支持的视频格式
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        
        # 创建输出目录
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'classified_files': {},
            'unknown_files': 0,
            'error_files': 0
        }
    
    def get_relative_path(self, file_path: Path) -> Path:
        """获取相对于输入路径的相对路径"""
        try:
            return file_path.relative_to(self.input_path)
        except ValueError:
            return file_path.name
    
    def create_output_dirs(self, detected_classes: List[str], relative_path: Path) -> List[Path]:
        """
        创建输出目录
        
        Args:
            detected_classes: 检测到的类别
            relative_path: 相对路径
            
        Returns:
            output_dirs: 输出目录列表
        """
        output_dirs = []
        
        if not detected_classes:
            # 没有检测到目标，保存到unknown文件夹
            unknown_dir = self.output_path / "unknown" / relative_path.parent
            unknown_dir.mkdir(parents=True, exist_ok=True)
            output_dirs.append(unknown_dir)
        else:
            # 为每个检测到的类别创建目录
            for class_name in detected_classes:
                class_dir = self.output_path / class_name / relative_path.parent
                class_dir.mkdir(parents=True, exist_ok=True)
                output_dirs.append(class_dir)
        
        return output_dirs
    
    def create_video_frame_dirs(self, detected_classes: List[str], video_name: str, relative_path: Path) -> List[Path]:
        """
        为视频帧创建输出目录
        
        Args:
            detected_classes: 检测到的类别
            video_name: 视频文件名（不含扩展名）
            relative_path: 相对路径
            
        Returns:
            output_dirs: 输出目录列表
        """
        output_dirs = []
        
        if not detected_classes:
            # 没有检测到目标，保存到unknown文件夹下的视频名文件夹
            unknown_dir = self.output_path / "unknown" / relative_path.parent / video_name
            unknown_dir.mkdir(parents=True, exist_ok=True)
            output_dirs.append(unknown_dir)
        else:
            # 为每个检测到的类别创建目录
            for class_name in detected_classes:
                class_dir = self.output_path / class_name / relative_path.parent / video_name
                class_dir.mkdir(parents=True, exist_ok=True)
                output_dirs.append(class_dir)
        
        return output_dirs
    
    def save_file_to_dirs(self, source_path: Path, output_dirs: List[Path], 
                         detected_classes: List[str]):
        """
        将文件保存到多个目录
        
        Args:
            source_path: 源文件路径
            output_dirs: 输出目录列表
            detected_classes: 检测到的类别
        """
        filename = source_path.name
        
        for i, output_dir in enumerate(output_dirs):
            dest_path = output_dir / filename
            
            try:
                # 复制原文件
                shutil.copy2(source_path, dest_path)
                
                # 更新统计信息
                if not detected_classes:
                    category = "unknown"
                    self.stats['unknown_files'] += 1
                else:
                    category = detected_classes[i] if i < len(detected_classes) else detected_classes[0]
                    if category not in self.stats['classified_files']:
                        self.stats['classified_files'][category] = 0
                    self.stats['classified_files'][category] += 1
                
                logger.info(f"已保存文件到 {category} 类别: {dest_path}")
                
            except Exception as e:
                logger.error(f"保存文件失败 {dest_path}: {e}")
                self.stats['error_files'] += 1
    
    def save_visualization(self, vis_image: np.ndarray, output_dirs: List[Path], 
                          filename: str, detected_classes: List[str]):
        """保存可视化结果到vis目录"""
        if not self.visualize:
            return
        
        # 为每个输出目录创建对应的vis目录
        for i, output_dir in enumerate(output_dirs):
            # 获取相对于输出根目录的路径
            try:
                relative_to_output = output_dir.relative_to(self.output_path)
                vis_dir = self.output_path / "vis" / relative_to_output
                vis_dir.mkdir(parents=True, exist_ok=True)
                
                vis_path = vis_dir / filename  # 不再添加"vis_"前缀
                cv2.imwrite(str(vis_path), vis_image)
                logger.info(f"已保存可视化结果: {vis_path}")
            except Exception as e:
                logger.error(f"保存可视化结果失败: {e}")
    
    def save_frame_to_dirs(self, frame: np.ndarray, output_dirs: List[Path], 
                          frame_filename: str, detected_classes: List[str]):
        """
        将视频帧保存到多个目录
        
        Args:
            frame: 视频帧
            output_dirs: 输出目录列表
            frame_filename: 帧文件名
            detected_classes: 检测到的类别
        """
        for i, output_dir in enumerate(output_dirs):
            frame_path = output_dir / frame_filename
            
            try:
                # 保存帧图片
                cv2.imwrite(str(frame_path), frame)
                
                # 更新统计信息
                if not detected_classes:
                    category = "unknown"
                    if 'frame_unknown_files' not in self.stats:
                        self.stats['frame_unknown_files'] = 0
                    self.stats['frame_unknown_files'] += 1
                else:
                    category = detected_classes[i] if i < len(detected_classes) else detected_classes[0]
                    if 'frame_classified_files' not in self.stats:
                        self.stats['frame_classified_files'] = {}
                    if category not in self.stats['frame_classified_files']:
                        self.stats['frame_classified_files'][category] = 0
                    self.stats['frame_classified_files'][category] += 1
                
            except Exception as e:
                logger.error(f"保存帧失败 {frame_path}: {e}")
                self.stats['error_files'] += 1
    
    def process_image(self, image_path: Path):
        """处理单张图像"""
        logger.info(f"正在处理图像: {image_path}")
        
        try:
            # 检测图像
            detected_classes, image, results_info = self.detector.detect_image(str(image_path))
            
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                self.stats['error_files'] += 1
                return
            
            # 获取相对路径
            relative_path = self.get_relative_path(image_path)
            
            # 创建输出目录
            output_dirs = self.create_output_dirs(detected_classes, relative_path)
            
            # 保存文件
            self.save_file_to_dirs(image_path, output_dirs, detected_classes)
            
            # 可视化（如果启用）
            if self.visualize and results_info['detections']:
                vis_image = self.detector.visualize_detection(image, results_info)
                self.save_visualization(vis_image, output_dirs, image_path.name, detected_classes)
            
            self.stats['processed_files'] += 1
            
        except Exception as e:
            logger.error(f"处理图像时发生错误 {image_path}: {e}")
            self.stats['error_files'] += 1
    
    def process_video(self, video_path: Path):
        """处理视频文件 - 逐帧推理并输出图片"""
        logger.info(f"正在处理视频: {video_path}")
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                self.stats['error_files'] += 1
                return
            
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"视频总帧数: {total_frames}, FPS: {fps}")
            
            # 获取相对路径和视频名称（不含扩展名）
            relative_path = self.get_relative_path(video_path)
            video_name = video_path.stem  # 不含扩展名的文件名
            
            frame_idx = 0
            processed_frames = 0
            
            # 逐帧处理
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测当前帧
                detected_classes, results_info = self.detector.detect_video_frame(frame)
                
                # 创建帧文件名（6位数字，左侧补零）
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                
                # 为每一帧创建输出目录（基于检测结果）
                output_dirs = self.create_video_frame_dirs(detected_classes, video_name, relative_path)
                
                # 保存当前帧
                self.save_frame_to_dirs(frame, output_dirs, frame_filename, detected_classes)
                
                # 可视化（如果启用）
                if self.visualize and results_info['detections']:
                    vis_image = self.detector.visualize_detection(frame, results_info)
                    
                    # 保存可视化帧到vis目录
                    for output_dir in output_dirs:
                        try:
                            relative_to_output = output_dir.relative_to(self.output_path)
                            vis_dir = self.output_path / "vis" / relative_to_output
                            vis_dir.mkdir(parents=True, exist_ok=True)
                            
                            vis_path = vis_dir / frame_filename  # 使用相同的文件名
                            cv2.imwrite(str(vis_path), vis_image)
                        except Exception as e:
                            logger.error(f"保存可视化帧失败: {e}")
                
                frame_idx += 1
                processed_frames += 1
                
                # 每处理100帧输出一次进度
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    logger.info(f"处理进度: {frame_idx}/{total_frames} ({progress:.1f}%)")
            
            cap.release()
            
            logger.info(f"视频处理完成，共处理 {processed_frames} 帧")
            self.stats['processed_files'] += 1
            
        except Exception as e:
            logger.error(f"处理视频时发生错误 {video_path}: {e}")
            self.stats['error_files'] += 1
    
    def process_all_files(self):
        """处理所有文件"""
        logger.info(f"开始处理目录: {self.input_path}")
        
        # 收集所有要处理的文件
        all_files = []
        
        if self.input_path.is_file():
            all_files.append(self.input_path)
        else:
            for file_path in self.input_path.rglob('*'):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in self.image_extensions or ext in self.video_extensions:
                        all_files.append(file_path)
        
        self.stats['total_files'] = len(all_files)
        logger.info(f"找到 {len(all_files)} 个文件需要处理")
        
        # 处理每个文件
        for file_path in all_files:
            ext = file_path.suffix.lower()
            
            if ext in self.image_extensions:
                self.process_image(file_path)
            elif ext in self.video_extensions:
                self.process_video(file_path)
        
        # 输出统计信息
        self.print_stats()
    
    def print_stats(self):
        """打印统计信息"""
        logger.info("=" * 50)
        logger.info("处理完成！统计信息:")
        logger.info(f"总文件数: {self.stats['total_files']}")
        logger.info(f"已处理: {self.stats['processed_files']}")
        logger.info(f"错误文件: {self.stats['error_files']}")
        logger.info(f"未检测到目标的文件: {self.stats['unknown_files']}")
        
        logger.info("各类别文件数:")
        for category, count in self.stats['classified_files'].items():
            logger.info(f"  {category}: {count}")
        
        # 输出视频帧统计信息
        if 'frame_classified_files' in self.stats or 'frame_unknown_files' in self.stats:
            logger.info("\n视频帧统计:")
            if 'frame_unknown_files' in self.stats:
                logger.info(f"未检测到目标的帧: {self.stats['frame_unknown_files']}")
            
            if 'frame_classified_files' in self.stats:
                logger.info("各类别帧数:")
                for category, count in self.stats['frame_classified_files'].items():
                    logger.info(f"  {category}: {count}")
        
        logger.info("=" * 50)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8目标检测和分类脚本')
    
    parser.add_argument('--input', '-i', required=True, 
                       help='输入路径（图像/视频文件或包含图像/视频的目录）')
    parser.add_argument('--output', '-o', required=True, 
                       help='输出路径（分类结果保存目录）')
    parser.add_argument('--model', '-m', required=True, 
                       help='YOLO模型文件路径（.pt文件）')
    parser.add_argument('--classes', '-c', required=True, 
                       help='类别文件路径（classes.txt）')
    parser.add_argument('--confidence', '-conf', type=float, default=0.5,
                       help='置信度阈值（默认: 0.5）')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='是否保存可视化检测结果')
    
    args = parser.parse_args()
    
    # 验证输入参数
    if not os.path.exists(args.input):
        logger.error(f"输入路径不存在: {args.input}")
        return
    
    if not os.path.exists(args.model):
        logger.error(f"模型文件不存在: {args.model}")
        return
    
    if not os.path.exists(args.classes):
        logger.error(f"类别文件不存在: {args.classes}")
        return
    
    try:
        # 初始化检测器
        detector = YOLODetectionClassifier(
            model_path=args.model,
            classes_path=args.classes,
            confidence_threshold=args.confidence
        )
        
        # 初始化文件处理器
        processor = FileProcessor(
            input_path=args.input,
            output_path=args.output,
            detector=detector,
            visualize=args.visualize
        )
        
        # 处理所有文件
        processor.process_all_files()
        
    except Exception as e:
        logger.error(f"程序执行时发生错误: {e}")
        raise

if __name__ == "__main__":
    main() 