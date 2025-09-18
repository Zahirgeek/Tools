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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from queue import Queue
import time
import torch # Added for device selection

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLODetectionClassifier:
    """YOLO检测分类器"""
    
    def __init__(self, model_path: str, classes_path: str, confidence_threshold: float = 0.5, device: str = "auto"):
        """
        初始化检测分类器
        
        Args:
            model_path: YOLO模型路径
            classes_path: 类别文件路径
            confidence_threshold: 置信度阈值
            device: 推理设备 ("auto", "cpu", "cuda", "0", "1" 等)
        """
        self.model_path = model_path
        self.classes_path = classes_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # 加载模型
        logger.info(f"正在加载YOLO模型: {model_path}")
        try:
            self.model = YOLO(model_path)
            # 设置推理设备
            if device == "cpu":
                self.model.to("cpu")
                logger.info("模型已设置为CPU推理模式")
            elif device == "auto":
                # 自动选择设备
                if torch.cuda.is_available():
                    self.model.to("cuda")
                    logger.info("模型已设置为CUDA推理模式")
                else:
                    self.model.to("cpu")
                    logger.info("CUDA不可用，模型已设置为CPU推理模式")
            else:
                # 用户指定的设备
                self.model.to(device)
                logger.info(f"模型已设置为设备: {device}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
        
        # 加载类别
        self.classes = self._load_classes(classes_path)
        logger.info(f"加载了 {len(self.classes)} 个类别: {self.classes}")
        logger.info(f"置信度阈值设置为: {self.confidence_threshold}")
        logger.info(f"推理设备: {self.model.device}")
    
    def _load_classes(self, classes_path: str) -> List[str]:
        """加载类别文件"""
        with open(classes_path, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return classes
    
    def detect_image(self, image_path: str, processor=None) -> Tuple[List[str], np.ndarray, dict, float]:
        """
        检测单张图像
        
        Args:
            image_path: 图像路径
            processor: 文件处理器（用于记录推理时间）
            
        Returns:
            detected_classes: 检测到的类别列表
            image: 原图像
            results_info: 检测结果信息
            inference_time: 推理时间（秒）
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return [], None, {}, 0.0
        
        # 记录推理阶段开始时间
        if processor:
            processor.record_inference_start()
        
        # 验证输入参数
        if not isinstance(self.confidence_threshold, (int, float)) or not np.isfinite(self.confidence_threshold):
            logger.error(f"置信度阈值无效: {self.confidence_threshold}")
            return [], image, {'detections': [], 'class_counts': {}}, 0.0
        
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            logger.warning(f"置信度阈值超出正常范围 [0,1]: {self.confidence_threshold}")
        
        # 进行检测并计时
        inference_start = time.time()
        try:
            results = self.model(image_path, conf=self.confidence_threshold)
        except Exception as e:
            logger.error(f"模型推理失败: {e}")
            return [], image, {'detections': [], 'class_counts': {}}, 0.0
        inference_time = time.time() - inference_start
        
        # 记录推理阶段结束时间
        if processor:
            processor.record_inference_end()
        
        detected_classes = []
        results_info = {
            'detections': [],
            'class_counts': {}
        }
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    try:
                        # 安全地提取类别ID和置信度
                        cls_tensor = box.cls
                        conf_tensor = box.conf
                        
                        if cls_tensor is None or conf_tensor is None:
                            logger.warning("跳过无效的检测框（类别或置信度为空）")
                            continue
                        
                        # 转换为numpy数组并验证
                        cls_array = cls_tensor.cpu().numpy()
                        conf_array = conf_tensor.cpu().numpy()
                        
                        if len(cls_array) == 0 or len(conf_array) == 0:
                            logger.warning("跳过空的检测框数据")
                            continue
                        
                        cls_id = int(cls_array[0])
                        confidence = float(conf_array[0])
                        
                        # 验证数据类型和范围
                        if not isinstance(cls_id, int) or not isinstance(confidence, (int, float)):
                            logger.warning(f"跳过无效的数据类型: cls_id={type(cls_id)}, confidence={type(confidence)}")
                            continue
                        
                        if not np.isfinite(cls_id) or not np.isfinite(confidence):
                            logger.warning(f"跳过非有限数值: cls_id={cls_id}, confidence={confidence}")
                            continue
                        
                        # 验证数值范围
                        if cls_id < 0 or not np.isfinite(confidence):
                            logger.warning(f"跳过无效的类别ID或置信度: cls_id={cls_id}, confidence={confidence}")
                            continue
                        
                        # 确保置信度满足阈值要求
                        if cls_id < len(self.classes) and confidence >= self.confidence_threshold:
                            class_name = self.classes[cls_id]
                            if class_name not in detected_classes:
                                detected_classes.append(class_name)
                            
                            # 记录检测信息
                            try:
                                bbox = box.xyxy.cpu().numpy()[0].tolist()
                                # 验证边界框数据
                                if len(bbox) == 4 and all(isinstance(x, (int, float)) and np.isfinite(x) for x in bbox):
                                    results_info['detections'].append({
                                        'class': class_name,
                                        'confidence': confidence,
                                        'bbox': bbox
                                    })
                                else:
                                    logger.warning(f"跳过无效的边界框数据: {bbox}")
                                    continue
                            except Exception as e:
                                logger.warning(f"处理边界框时出错: {e}")
                                continue
                            
                            # 统计类别数量
                            if class_name not in results_info['class_counts']:
                                results_info['class_counts'][class_name] = 0
                            results_info['class_counts'][class_name] += 1
                        elif cls_id < len(self.classes):
                            # 记录被过滤掉的低置信度检测
                            class_name = self.classes[cls_id]
                            logger.debug(f"过滤低置信度检测: {class_name} (置信度: {confidence:.3f} < {self.confidence_threshold:.3f})")
                        else:
                            logger.warning(f"类别ID超出范围: {cls_id} >= {len(self.classes)}")
                            
                    except Exception as e:
                        logger.warning(f"处理检测框时出错: {e}")
                        continue
        
        return detected_classes, image, results_info, inference_time
    
    def detect_images_batch(self, image_paths: List[str], batch_size: int = 8, processor=None) -> List[Tuple[str, List[str], np.ndarray, dict, float]]:
        """
        批量检测多张图像
        
        Args:
            image_paths: 图像路径列表
            batch_size: 批处理大小
            processor: 文件处理器（用于记录推理时间）
            
        Returns:
            results: 检测结果列表，每个元素为(image_path, detected_classes, image, results_info, inference_time)
        """
        all_results = []
        
        # 分批处理
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # 读取批量图像
            batch_images = []
            valid_paths = []
            
            for path in batch_paths:
                image = cv2.imread(path)
                if image is not None:
                    batch_images.append(image)
                    valid_paths.append(path)
                else:
                    logger.error(f"无法读取图像: {path}")
                    all_results.append((path, [], None, {}, 0.0))
            
            if not batch_images:
                continue
            
            # 记录推理阶段开始时间
            if processor:
                processor.record_inference_start()
            
            # 验证输入参数
            if not isinstance(self.confidence_threshold, (int, float)) or not np.isfinite(self.confidence_threshold):
                logger.error(f"置信度阈值无效: {self.confidence_threshold}")
                # 为每个无效路径返回空结果
                for path in valid_paths:
                    all_results.append((path, [], None, {'detections': [], 'class_counts': {}}, 0.0))
                continue
            
            if self.confidence_threshold < 0 or self.confidence_threshold > 1:
                logger.warning(f"置信度阈值超出正常范围 [0,1]: {self.confidence_threshold}")
            
            # 批量推理
            try:
                batch_inference_start = time.time()
                results = self.model(batch_images, conf=self.confidence_threshold)
                batch_inference_time = time.time() - batch_inference_start
                avg_inference_time = batch_inference_time / len(batch_images)
                
                # 记录推理阶段结束时间
                if processor:
                    processor.record_inference_end()
                
                # 处理每个结果
                for j, (path, image, result) in enumerate(zip(valid_paths, batch_images, results)):
                    detected_classes = []
                    results_info = {
                        'detections': [],
                        'class_counts': {}
                    }
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            try:
                                # 安全地提取类别ID和置信度
                                cls_tensor = box.cls
                                conf_tensor = box.conf
                                
                                if cls_tensor is None or conf_tensor is None:
                                    logger.warning("跳过无效的检测框（类别或置信度为空）")
                                    continue
                                
                                # 转换为numpy数组并验证
                                cls_array = cls_tensor.cpu().numpy()
                                conf_array = conf_tensor.cpu().numpy()
                                
                                if len(cls_array) == 0 or len(conf_array) == 0:
                                    logger.warning("跳过空的检测框数据")
                                    continue
                                
                                cls_id = int(cls_array[0])
                                confidence = float(conf_array[0])
                                
                                # 验证数据类型和范围
                                if not isinstance(cls_id, int) or not isinstance(confidence, (int, float)):
                                    logger.warning(f"跳过无效的数据类型: cls_id={type(cls_id)}, confidence={type(confidence)}")
                                    continue
                                
                                if not np.isfinite(cls_id) or not np.isfinite(confidence):
                                    logger.warning(f"跳过非有限数值: cls_id={cls_id}, confidence={confidence}")
                                    continue
                                
                                # 验证数值范围
                                if cls_id < 0 or not np.isfinite(confidence):
                                    logger.warning(f"跳过无效的类别ID或置信度: cls_id={cls_id}, confidence={confidence}")
                                    continue
                                
                                # 确保置信度满足阈值要求
                                if cls_id < len(self.classes) and confidence >= self.confidence_threshold:
                                    class_name = self.classes[cls_id]
                                    if class_name not in detected_classes:
                                        detected_classes.append(class_name)
                                    
                                    # 记录检测信息
                                    try:
                                        bbox = box.xyxy.cpu().numpy()[0].tolist()
                                        # 验证边界框数据
                                        if len(bbox) == 4 and all(isinstance(x, (int, float)) and np.isfinite(x) for x in bbox):
                                            results_info['detections'].append({
                                                'class': class_name,
                                                'confidence': confidence,
                                                'bbox': bbox
                                            })
                                        else:
                                            logger.warning(f"跳过无效的边界框数据: {bbox}")
                                            continue
                                    except Exception as e:
                                        logger.warning(f"处理边界框时出错: {e}")
                                        continue
                                    
                                    # 统计类别数量
                                    if class_name not in results_info['class_counts']:
                                        results_info['class_counts'][class_name] = 0
                                    results_info['class_counts'][class_name] += 1
                                elif cls_id < len(self.classes):
                                    # 记录被过滤掉的低置信度检测
                                    class_name = self.classes[cls_id]
                                    logger.debug(f"过滤低置信度检测: {class_name} (置信度: {confidence:.3f} < {self.confidence_threshold:.3f})")
                                else:
                                    logger.warning(f"类别ID超出范围: {cls_id} >= {len(self.classes)}")
                                    
                            except Exception as e:
                                logger.warning(f"处理检测框时出错: {e}")
                                continue
                    
                    all_results.append((path, detected_classes, image, results_info, avg_inference_time))
                    
            except Exception as e:
                logger.error(f"批量推理失败: {e}")
                # 失败时回退到单张处理
                for path in valid_paths:
                    try:
                        detected_classes, image, results_info, inference_time = self.detect_image(path, processor)
                        all_results.append((path, detected_classes, image, results_info, inference_time))
                    except Exception as e2:
                        logger.error(f"单张图像处理失败 {path}: {e2}")
                        all_results.append((path, [], None, {}, 0.0))
        
        return all_results
    
    def detect_video_frame(self, frame: np.ndarray, processor=None) -> Tuple[List[str], dict, float]:
        """
        检测视频帧
        
        Args:
            frame: 视频帧
            processor: 文件处理器（用于记录推理时间）
            
        Returns:
            detected_classes: 检测到的类别列表
            results_info: 检测结果信息
            inference_time: 推理时间（秒）
        """
        # 记录推理阶段开始时间
        if processor:
            processor.record_inference_start()
        
        # 验证输入参数
        if not isinstance(self.confidence_threshold, (int, float)) or not np.isfinite(self.confidence_threshold):
            logger.error(f"置信度阈值无效: {self.confidence_threshold}")
            return [], {'detections': [], 'class_counts': {}}, 0.0
        
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            logger.warning(f"置信度阈值超出正常范围 [0,1]: {self.confidence_threshold}")
        
        # 验证输入帧
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error("输入帧无效")
            return [], {'detections': [], 'class_counts': {}}, 0.0
        
        inference_start = time.time()
        try:
            results = self.model(frame, conf=self.confidence_threshold)
        except Exception as e:
            logger.error(f"模型推理失败: {e}")
            return [], {'detections': [], 'class_counts': {}}, 0.0
        inference_time = time.time() - inference_start
        
        # 记录推理阶段结束时间
        if processor:
            processor.record_inference_end()
        
        detected_classes = []
        results_info = {
            'detections': [],
            'class_counts': {}
        }
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    try:
                        # 安全地提取类别ID和置信度
                        cls_tensor = box.cls
                        conf_tensor = box.conf
                        
                        if cls_tensor is None or conf_tensor is None:
                            logger.warning("跳过无效的检测框（类别或置信度为空）")
                            continue
                        
                        # 转换为numpy数组并验证
                        cls_array = cls_tensor.cpu().numpy()
                        conf_array = conf_tensor.cpu().numpy()
                        
                        if len(cls_array) == 0 or len(conf_array) == 0:
                            logger.warning("跳过空的检测框数据")
                            continue
                        
                        cls_id = int(cls_array[0])
                        confidence = float(conf_array[0])
                        
                        # 验证数据类型和范围
                        if not isinstance(cls_id, int) or not isinstance(confidence, (int, float)):
                            logger.warning(f"跳过无效的数据类型: cls_id={type(cls_id)}, confidence={type(confidence)}")
                            continue
                        
                        if not np.isfinite(cls_id) or not np.isfinite(confidence):
                            logger.warning(f"跳过非有限数值: cls_id={cls_id}, confidence={confidence}")
                            continue
                        
                        # 验证数值范围
                        if cls_id < 0 or not np.isfinite(confidence):
                            logger.warning(f"跳过无效的类别ID或置信度: cls_id={cls_id}, confidence={confidence}")
                            continue
                        
                        # 确保置信度满足阈值要求
                        if cls_id < len(self.classes) and confidence >= self.confidence_threshold:
                            class_name = self.classes[cls_id]
                            if class_name not in detected_classes:
                                detected_classes.append(class_name)
                            
                            # 记录检测信息
                            try:
                                bbox = box.xyxy.cpu().numpy()[0].tolist()
                                # 验证边界框数据
                                if len(bbox) == 4 and all(isinstance(x, (int, float)) and np.isfinite(x) for x in bbox):
                                    results_info['detections'].append({
                                        'class': class_name,
                                        'confidence': confidence,
                                        'bbox': bbox
                                    })
                                else:
                                    logger.warning(f"跳过无效的边界框数据: {bbox}")
                                    continue
                            except Exception as e:
                                logger.warning(f"处理边界框时出错: {e}")
                                continue
                            
                            # 统计类别数量
                            if class_name not in results_info['class_counts']:
                                results_info['class_counts'][class_name] = 0
                            results_info['class_counts'][class_name] += 1
                        elif cls_id < len(self.classes):
                            # 记录被过滤掉的低置信度检测
                            class_name = self.classes[cls_id]
                            logger.debug(f"过滤低置信度检测: {class_name} (置信度: {confidence:.3f} < {self.confidence_threshold:.3f})")
                        else:
                            logger.warning(f"类别ID超出范围: {cls_id} >= {len(self.classes)}")
                            
                    except Exception as e:
                        logger.warning(f"处理检测框时出错: {e}")
                        continue
        
        return detected_classes, results_info, inference_time
    
    def visualize_detection(self, image: np.ndarray, results_info: dict, target_class: str = None) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原图像
            results_info: 检测结果信息
            target_class: 目标类别，如果指定则只显示该类别的检测框
            
        Returns:
            visualized_image: 可视化后的图像
        """
        vis_image = image.copy()
        height, width = vis_image.shape[:2]
        
        for detection in results_info['detections']:
            class_name = detection['class']
            confidence = detection['confidence']
            
            # 如果指定了目标类别，则只显示该类别的检测框
            if target_class is not None and class_name != target_class:
                continue
            
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # 确保边界框坐标在图片范围内
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            
            # 获取文字尺寸
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 计算标签背景框的位置
            label_x = x1
            label_y = y1 - 10  # 在边界框上方留10像素间距
            
            # 如果标签背景框超出上边界，则放在边界框下方
            if label_y - text_height - baseline < 0:
                label_y = y2 + text_height + baseline + 10
            
            # 如果标签背景框超出下边界，则放在边界框内部上方
            if label_y + baseline > height:
                label_y = y1 + text_height + baseline + 10
            
            # 确保标签背景框不超出左右边界
            if label_x + text_width > width:
                label_x = width - text_width - 5
            
            if label_x < 0:
                label_x = 5
            
            # 绘制标签背景框
            cv2.rectangle(vis_image, 
                         (label_x - 5, label_y - text_height - baseline - 5),
                         (label_x + text_width + 5, label_y + baseline + 5),
                         (0, 255, 0), -1)
            
            # 绘制标签文字
            cv2.putText(vis_image, label, (label_x, label_y), 
                       font, font_scale, (0, 0, 0), thickness)
        
        return vis_image

class FileProcessor:
    """文件处理器"""
    
    def __init__(self, input_path: str, output_path: str, detector: YOLODetectionClassifier, 
                 visualize: bool = False, max_workers: int = 4, batch_size: int = 8, save_labels: bool = False, 
                 single_folder: bool = False):
        """
        初始化文件处理器
        
        Args:
            input_path: 输入路径
            output_path: 输出路径
            detector: YOLO检测器
            visualize: 是否可视化
            max_workers: 最大并行工作线程数
            batch_size: 图像批处理大小
            save_labels: 是否保存YOLO格式标签
            single_folder: 是否将所有结果保存在单个文件夹中（不按类别分类）
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.detector = detector
        self.visualize = visualize
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.save_labels = save_labels
        self.single_folder = single_folder
        
        # 支持的图像格式
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        # 支持的视频格式
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        
        # 创建输出目录
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 统计信息和线程锁
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'classified_files': {},
            'unknown_files': 0,
            'error_files': 0,
            'total_inference_time': 0.0,  # 累计推理时间（用于计算平均）
            'total_inferences': 0,        # 总推理次数
        }
        self.stats_lock = threading.Lock()
        
        # 进度显示
        self.progress_count = 0
        self.start_time = None
        self.end_time = None
        
        # 推理阶段时间统计
        self.inference_start_time = None  # 第一次推理开始时间
        self.inference_end_time = None    # 最后一次推理结束时间
    
    def get_relative_path(self, file_path: Path) -> Path:
        """获取相对于输入路径的相对路径"""
        try:
            return file_path.relative_to(self.input_path)
        except ValueError:
            return file_path.name
    
    def record_inference_start(self):
        """记录推理阶段开始时间（线程安全）"""
        with self.stats_lock:
            if self.inference_start_time is None:
                self.inference_start_time = time.time()
    
    def record_inference_end(self):
        """记录推理阶段结束时间（线程安全）"""
        with self.stats_lock:
            self.inference_end_time = time.time()
    
    def update_stats_thread_safe(self, detected_classes: List[str], is_frame: bool = False, inference_time: float = 0.0):
        """线程安全地更新统计信息"""
        with self.stats_lock:
            self.progress_count += 1
            
            # 记录推理时间和次数
            if inference_time > 0:
                self.stats['total_inference_time'] += inference_time
                self.stats['total_inferences'] += 1
            
            if not detected_classes:
                if is_frame:
                    if 'frame_unknown_files' not in self.stats:
                        self.stats['frame_unknown_files'] = 0
                    self.stats['frame_unknown_files'] += 1
                else:
                    self.stats['unknown_files'] += 1
            else:
                # 只统计第一个检测到的类别以避免重复计数
                category = detected_classes[0]
                if is_frame:
                    if 'frame_classified_files' not in self.stats:
                        self.stats['frame_classified_files'] = {}
                    if category not in self.stats['frame_classified_files']:
                        self.stats['frame_classified_files'][category] = 0
                    self.stats['frame_classified_files'][category] += 1
                else:
                    if category not in self.stats['classified_files']:
                        self.stats['classified_files'][category] = 0
                    self.stats['classified_files'][category] += 1
    
    def update_processed_files(self):
        """更新已处理文件数"""
        with self.stats_lock:
            self.stats['processed_files'] += 1
    
    def update_error_files(self):
        """更新错误文件数"""
        with self.stats_lock:
            self.stats['error_files'] += 1
    
    def show_progress(self, file_path: str = None):
        """显示处理进度"""
        if self.start_time is None:
            return
            
        elapsed_time = time.time() - self.start_time
        with self.stats_lock:
            progress = self.progress_count
            total = self.stats['total_files']
            
        if total > 0:
            percentage = (progress / total) * 100
            avg_time = elapsed_time / progress if progress > 0 else 0
            eta = avg_time * (total - progress) if progress > 0 else 0
            
            if file_path:
                logger.info(f"进度: {progress}/{total} ({percentage:.1f}%) | "
                           f"平均: {avg_time:.2f}s/文件 | "
                           f"预计剩余: {eta:.0f}s | "
                           f"当前: {Path(file_path).name}")
            else:
                logger.info(f"进度: {progress}/{total} ({percentage:.1f}%) | "
                           f"已用时: {elapsed_time:.0f}s")
    
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
        
        if self.single_folder:
            # 单文件夹模式：所有结果都保存在一个文件夹中
            single_dir = self.output_path / "detected" / relative_path.parent
            single_dir.mkdir(parents=True, exist_ok=True)
            output_dirs.append(single_dir)
        else:
            # 原模式：按类别分文件夹
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
        
        if self.single_folder:
            # 单文件夹模式：所有视频帧都保存在一个文件夹中
            single_dir = self.output_path / "detected" / relative_path.parent / video_name
            single_dir.mkdir(parents=True, exist_ok=True)
            output_dirs.append(single_dir)
        else:
            # 原模式：按类别分文件夹
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
                
                category = "unknown" if not detected_classes else (
                    detected_classes[i] if i < len(detected_classes) else detected_classes[0]
                )
                
            except Exception as e:
                logger.error(f"保存文件失败 {dest_path}: {e}")
                self.update_error_files()
    
    def save_visualization(self, image: np.ndarray, results_info: dict, output_dirs: List[Path], 
                          filename: str, detected_classes: List[str]):
        """保存可视化结果到vis目录"""
        if not self.visualize:
            return
        
        if self.single_folder:
            # 单文件夹模式：只保存一个可视化图片，显示所有检测框
            try:
                vis_dir = self.output_path / "vis" / "detected"
                vis_dir.mkdir(parents=True, exist_ok=True)
                
                # 显示所有检测框
                vis_image = self.detector.visualize_detection(image, results_info, target_class=None)
                
                vis_path = vis_dir / filename
                cv2.imwrite(str(vis_path), vis_image)
                logger.info(f"已保存可视化结果: {vis_path} (显示所有检测框)")
            except Exception as e:
                logger.error(f"保存可视化结果失败: {e}")
        else:
            # 原模式：为每个输出目录创建对应的vis目录，每个标签目录下只显示该标签的检测框
            for i, output_dir in enumerate(output_dirs):
                # 获取相对于输出根目录的路径
                try:
                    relative_to_output = output_dir.relative_to(self.output_path)
                    vis_dir = self.output_path / "vis" / relative_to_output
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 确定当前目录对应的标签类别
                    if not detected_classes:
                        # 如果是unknown目录，显示所有检测框
                        target_class = None
                    else:
                        # 根据目录名确定目标类别
                        # 从目录路径中提取类别名（最后一个目录名）
                        target_class = detected_classes[i] if i < len(detected_classes) else detected_classes[0]
                    
                    # 生成只包含目标类别检测框的可视化图片
                    vis_image = self.detector.visualize_detection(image, results_info, target_class)
                    
                    vis_path = vis_dir / filename
                    cv2.imwrite(str(vis_path), vis_image)
                    logger.info(f"已保存可视化结果: {vis_path} (类别: {target_class if target_class else 'unknown'})")
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
                
            except Exception as e:
                logger.error(f"保存帧失败 {frame_path}: {e}")
                self.update_error_files()
    
    def save_yolo_label(self, image_path: Path, detections: list, image_shape: tuple, image_data: np.ndarray = None, video_name: str = None, label_filename: str = None):
        """
        保存YOLO格式标签文件和对应的图片到yolo_dataset目录
        Args:
            image_path: 原图像路径（Path对象）
            detections: 检测结果列表，每个元素为dict，包含class/confidence/bbox
            image_shape: (height, width)
            image_data: 图像数据（numpy数组）
            video_name: 视频文件名（如有，表示为视频帧标签）
            label_filename: 指定标签文件名（如有，优先使用）
        """
        if not detections:
            return
        
        # 计算相对路径，构造标签和图片文件路径
        relative_path = self.get_relative_path(image_path)
        
        # 创建yolo_dataset目录结构
        if self.single_folder:
            # 单文件夹模式：所有标签和图片都保存在统一位置，不按类别分类
            if video_name:
                # 视频帧的情况
                yolo_images_dir = self.output_path / "yolo_dataset" / "images" / video_name
                yolo_labels_dir = self.output_path / "yolo_dataset" / "labels" / video_name
            else:
                # 图片的情况
                yolo_images_dir = self.output_path / "yolo_dataset" / "images"
                yolo_labels_dir = self.output_path / "yolo_dataset" / "labels"
        else:
            # 原模式：按类别分文件夹
            if video_name:
                # 视频帧的情况
                yolo_images_dir = self.output_path / "yolo_dataset" / "images" / relative_path.parent / video_name
                yolo_labels_dir = self.output_path / "yolo_dataset" / "labels" / relative_path.parent / video_name
            else:
                # 图片的情况
                yolo_images_dir = self.output_path / "yolo_dataset" / "images" / relative_path.parent
                yolo_labels_dir = self.output_path / "yolo_dataset" / "labels" / relative_path.parent
        
        # 创建目录
        yolo_images_dir.mkdir(parents=True, exist_ok=True)
        yolo_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 确定文件名
        if label_filename:
            # 视频帧的情况，使用指定的文件名
            base_name = label_filename.replace('.txt', '')
            image_filename = base_name + '.jpg'
            label_filename_final = label_filename
        else:
            # 图片的情况，使用原文件名
            base_name = image_path.stem
            image_filename = image_path.name
            label_filename_final = base_name + '.txt'
        
        # 保存图片
        if image_data is not None:
            image_path_yolo = yolo_images_dir / image_filename
            try:
                cv2.imwrite(str(image_path_yolo), image_data)
                logger.info(f"已保存YOLO图片: {image_path_yolo}")
            except Exception as e:
                logger.error(f"保存YOLO图片失败: {e}")
        else:
            # 如果没有提供图像数据，复制原文件
            image_path_yolo = yolo_images_dir / image_filename
            try:
                shutil.copy2(image_path, image_path_yolo)
                logger.info(f"已复制YOLO图片: {image_path_yolo}")
            except Exception as e:
                logger.error(f"复制YOLO图片失败: {e}")
        
        # 保存标签文件
        label_path = yolo_labels_dir / label_filename_final
        h, w = image_shape[:2]
        lines = []
        for det in detections:
            class_name = det['class']
            if class_name in self.detector.classes:
                class_id = self.detector.classes.index(class_name)
            else:
                continue
            x1, y1, x2, y2 = det['bbox']
            # 归一化
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # 写入标签文件
        try:
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            logger.info(f"已保存YOLO标签: {label_path}")
        except Exception as e:
            logger.error(f"保存YOLO标签失败: {e}")
    
    def process_images_batch(self, image_paths: List[Path]) -> None:
        """
        批量处理图像文件
        
        Args:
            image_paths: 图像路径列表
        """
        try:
            # 转换为字符串路径
            str_paths = [str(path) for path in image_paths]
            
            # 批量检测
            batch_results = self.detector.detect_images_batch(str_paths, self.batch_size, self)
            
            # 处理每个结果
            for image_path_str, detected_classes, image, results_info, inference_time in batch_results:
                image_path = Path(image_path_str)
                
                try:
                    if image is None:
                        logger.error(f"无法读取图像: {image_path}")
                        self.update_error_files()
                        continue
                    
                    # 获取相对路径
                    relative_path = self.get_relative_path(image_path)
                    
                    # 创建输出目录
                    output_dirs = self.create_output_dirs(detected_classes, relative_path)
                    
                    # 保存文件
                    self.save_file_to_dirs(image_path, output_dirs, detected_classes)
                    
                    # 可视化（如果启用）
                    if self.visualize and results_info['detections']:
                        self.save_visualization(image, results_info, output_dirs, image_path.name, detected_classes)
                    
                    # 保存YOLO标签（如果启用）
                    if self.save_labels and results_info['detections']:
                        self.save_yolo_label(image_path, results_info['detections'], image.shape, image_data=image)
                    
                    # 更新统计信息
                    self.update_stats_thread_safe(detected_classes, is_frame=False, inference_time=inference_time)
                    
                except Exception as e:
                    logger.error(f"处理图像时发生错误 {image_path}: {e}")
                    self.update_error_files()
            
            # 更新已处理文件数
            self.update_processed_files()
            
        except Exception as e:
            logger.error(f"批量处理图像时发生错误: {e}")
            # 回退到单张处理
            for image_path in image_paths:
                self.process_image(image_path)
    
    def process_video_worker(self, video_path: Path) -> None:
        """
        工作线程处理单个视频文件
        
        Args:
            video_path: 视频文件路径
        """
        try:
            self.process_video(video_path)
            self.update_processed_files()
        except Exception as e:
            logger.error(f"处理视频时发生错误 {video_path}: {e}")
            self.update_error_files()
    
    def process_image(self, image_path: Path):
        """处理单张图像（用于回退处理）"""
        try:
            # 检测图像
            detected_classes, image, results_info, inference_time = self.detector.detect_image(str(image_path), self)
            
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                self.update_error_files()
                return
            
            # 获取相对路径
            relative_path = self.get_relative_path(image_path)
            
            # 创建输出目录
            output_dirs = self.create_output_dirs(detected_classes, relative_path)
            
            # 保存文件
            self.save_file_to_dirs(image_path, output_dirs, detected_classes)
            
            # 可视化（如果启用）
            if self.visualize and results_info['detections']:
                self.save_visualization(image, results_info, output_dirs, image_path.name, detected_classes)
            
            # 保存YOLO标签（如果启用）
            if self.save_labels and results_info['detections']:
                self.save_yolo_label(image_path, results_info['detections'], image.shape, image_data=image)
            
            # 更新统计信息
            self.update_stats_thread_safe(detected_classes, is_frame=False, inference_time=inference_time)
            
        except Exception as e:
            logger.error(f"处理图像时发生错误 {image_path}: {e}")
            self.update_error_files()
    
    def process_video(self, video_path: Path):
        """处理视频文件 - 逐帧推理并输出图片"""
        logger.info(f"正在处理视频: {video_path}")
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                self.update_error_files()
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
                detected_classes, results_info, inference_time = self.detector.detect_video_frame(frame, self)
                
                # 创建帧文件名（6位数字，左侧补零）
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                label_filename = f"frame_{frame_idx:06d}.txt"
                
                # 为每一帧创建输出目录（基于检测结果）
                output_dirs = self.create_video_frame_dirs(detected_classes, video_name, relative_path)
                
                # 保存当前帧
                self.save_frame_to_dirs(frame, output_dirs, frame_filename, detected_classes)
                
                # 更新帧统计信息
                self.update_stats_thread_safe(detected_classes, is_frame=True, inference_time=inference_time)
                
                # 可视化（如果启用）
                if self.visualize and results_info['detections']:
                    if self.single_folder:
                        # 单文件夹模式：只保存一个可视化图片，显示所有检测框
                        try:
                            vis_dir = self.output_path / "vis" / "detected" / relative_path.parent / video_name
                            vis_dir.mkdir(parents=True, exist_ok=True)
                            
                            # 显示所有检测框
                            vis_image = self.detector.visualize_detection(frame, results_info, target_class=None)
                            
                            vis_path = vis_dir / frame_filename
                            cv2.imwrite(str(vis_path), vis_image)
                        except Exception as e:
                            logger.error(f"保存可视化帧失败: {e}")
                    else:
                        # 原模式：为每个输出目录创建对应的vis目录并保存对应的可视化结果
                        for i, output_dir in enumerate(output_dirs):
                            try:
                                relative_to_output = output_dir.relative_to(self.output_path)
                                vis_dir = self.output_path / "vis" / relative_to_output
                                vis_dir.mkdir(parents=True, exist_ok=True)
                                
                                # 确定当前目录对应的标签类别
                                if not detected_classes:
                                    # 如果是unknown目录，显示所有检测框
                                    target_class = None
                                else:
                                    # 根据目录名确定目标类别
                                    target_class = detected_classes[i] if i < len(detected_classes) else detected_classes[0]
                                
                                # 生成只包含目标类别检测框的可视化图片
                                vis_image = self.detector.visualize_detection(frame, results_info, target_class)
                                
                                vis_path = vis_dir / frame_filename
                                cv2.imwrite(str(vis_path), vis_image)
                            except Exception as e:
                                logger.error(f"保存可视化帧失败: {e}")
                
                # 保存YOLO标签（如果启用）
                if self.save_labels and results_info['detections']:
                    self.save_yolo_label(video_path, results_info['detections'], frame.shape, image_data=frame, video_name=video_name, label_filename=label_filename)
                
                frame_idx += 1
                processed_frames += 1
                
                # 每处理100帧输出一次进度
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    logger.info(f"视频 {video_path.name} 进度: {frame_idx}/{total_frames} ({progress:.1f}%)")
            
            cap.release()
            
            logger.info(f"视频处理完成: {video_path.name}，共处理 {processed_frames} 帧")
            
        except Exception as e:
            logger.error(f"处理视频时发生错误 {video_path}: {e}")
            self.update_error_files()
    
    def process_all_files(self):
        """智能处理所有文件 - 根据文件类型和数量优化策略"""
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
        
        # 分离图像和视频文件
        image_files = []
        video_files = []
        
        for file_path in all_files:
            ext = file_path.suffix.lower()
            if ext in self.image_extensions:
                image_files.append(file_path)
            elif ext in self.video_extensions:
                video_files.append(file_path)
        
        self.stats['total_files'] = len(all_files)
        logger.info(f"找到 {len(all_files)} 个文件需要处理:")
        logger.info(f"  图像文件: {len(image_files)} 个")
        logger.info(f"  视频文件: {len(video_files)} 个")
        
        # 🚀 智能策略选择
        optimal_strategy = self._choose_optimal_strategy(image_files, video_files)
        logger.info(f"  采用策略: {optimal_strategy['name']}")
        logger.info(f"  并行度: {optimal_strategy['workers']} 线程")
        logger.info(f"  批处理大小: {optimal_strategy['batch_size']}")
        
        if not all_files:
            logger.warning("未找到要处理的文件")
            return
        
        self.start_time = time.time()
        
        # 根据选择的策略执行处理
        if optimal_strategy['strategy'] == 'batch_only':
            self._process_batch_only(image_files, video_files, optimal_strategy)
        else:
            self._process_with_threads(image_files, video_files, optimal_strategy)
        
        # 显示最终进度
        self.show_progress()
        
        # 设置结束时间
        self.end_time = time.time()
        
        # 输出统计信息
        self.print_stats()
    
    def _choose_optimal_strategy(self, image_files, video_files):
        """根据文件数量和类型选择最优处理策略"""
        total_images = len(image_files)
        total_videos = len(video_files)
        
        # 策略1: 纯批量处理（适合大量图像，少量或无视频）
        if total_images >= 100 and total_videos <= 2:
            return {
                'strategy': 'batch_only',
                'name': '批量优化模式（推荐）',
                'workers': 1,  # 使用单线程避免GPU竞争
                'batch_size': min(32, max(8, total_images // 10))  # 动态批大小
            }
        
        # 策略2: 小规模处理（文件数量少）
        elif total_images + total_videos <= 50:
            return {
                'strategy': 'single_thread',
                'name': '单线程模式（小规模）',
                'workers': 1,
                'batch_size': min(8, total_images)
            }
        
        # 策略3: 混合并行处理（图像和视频都较多）
        else:
            # 动态调整线程数，避免过度并行
            optimal_workers = min(self.max_workers, 
                                max(1, min(4, total_videos + (total_images // 50))))
            return {
                'strategy': 'hybrid_parallel',
                'name': '混合并行模式',
                'workers': optimal_workers,
                'batch_size': min(16, max(4, total_images // optimal_workers))
            }
    
    def _process_batch_only(self, image_files, video_files, strategy):
        """批量优化处理模式"""
        logger.info("🚀 使用批量优化模式处理")
        
        # 处理图像文件 - 使用大批量
        if image_files:
            batch_size = strategy['batch_size']
            total_batches = (len(image_files) + batch_size - 1) // batch_size
            
            for i in range(0, len(image_files), batch_size):
                batch = image_files[i:i + batch_size]
                batch_num = i // batch_size + 1
                logger.info(f"处理图像批次 {batch_num}/{total_batches}: {len(batch)} 个文件")
                
                self.process_images_batch(batch)
                
                # 显示批次进度
                if batch_num % 5 == 0 or batch_num == total_batches:
                    progress = (batch_num / total_batches) * 100
                    logger.info(f"图像处理进度: {progress:.1f}%")
        
        # 处理视频文件 - 串行处理避免竞争
        for i, video_file in enumerate(video_files):
            logger.info(f"处理视频 {i+1}/{len(video_files)}: {video_file.name}")
            self.process_video_worker(video_file)
    
    def _process_with_threads(self, image_files, video_files, strategy):
        """多线程处理模式"""
        logger.info(f"🔄 使用多线程模式处理 ({strategy['workers']} 线程)")
        
        # 创建线程池处理
        with ThreadPoolExecutor(max_workers=strategy['workers']) as executor:
            futures = []
            
            # 提交图像批处理任务
            if image_files:
                batch_size = strategy['batch_size']
                # 将图像文件分批
                for i in range(0, len(image_files), batch_size):
                    batch = image_files[i:i + batch_size]
                    future = executor.submit(self.process_images_batch, batch)
                    futures.append(future)
                    logger.info(f"提交图像批处理任务: {len(batch)} 个文件")
            
            # 提交视频处理任务
            for video_file in video_files:
                future = executor.submit(self.process_video_worker, video_file)
                futures.append(future)
                logger.info(f"提交视频处理任务: {video_file.name}")
            
            # 等待所有任务完成并显示进度
            completed_tasks = 0
            total_tasks = len(futures)
            
            for future in as_completed(futures):
                try:
                    future.result()  # 获取结果以捕获异常
                    completed_tasks += 1
                    
                    # 显示任务完成进度
                    task_progress = (completed_tasks / total_tasks) * 100
                    logger.info(f"任务进度: {completed_tasks}/{total_tasks} ({task_progress:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"任务执行失败: {e}")
                    completed_tasks += 1
    
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
        
        # 输出总处理时间和推理时间
        if self.start_time and self.end_time:
            total_processing_time = self.end_time - self.start_time
            logger.info(f"\n总处理时间: {total_processing_time:.2f} 秒")
            
            # 推理阶段总时长（真实的推理时间段）
            if self.inference_start_time and self.inference_end_time:
                inference_phase_duration = self.inference_end_time - self.inference_start_time
                logger.info(f"推理阶段总时长: {inference_phase_duration:.3f} 秒")
                logger.info(f"推理时间占比: {inference_phase_duration / total_processing_time * 100:.1f}%")
            else:
                logger.info("推理阶段总时长: 无法计算（无推理操作）")
            
            # 累计推理时间（用于计算平均推理时间）
            logger.info(f"累计推理时间: {self.stats['total_inference_time']:.3f} 秒")
            
            # 计算处理的总项目数（文件 + 帧）
            total_processed_items = self.stats['processed_files']
            if 'frame_classified_files' in self.stats:
                total_processed_items += sum(self.stats['frame_classified_files'].values())
            if 'frame_unknown_files' in self.stats:
                total_processed_items += self.stats['frame_unknown_files']
            
            if self.stats['processed_files'] > 0:
                logger.info(f"平均每文件处理时间: {total_processing_time / self.stats['processed_files']:.3f} 秒")
            
            if self.stats['total_inferences'] > 0:
                logger.info(f"总推理次数: {self.stats['total_inferences']}")
                logger.info(f"平均单次推理时间: {self.stats['total_inference_time'] / self.stats['total_inferences'] * 1000:.1f} 毫秒")
                
                # 使用推理阶段总时长计算实际FPS
                if self.inference_start_time and self.inference_end_time:
                    inference_phase_duration = self.inference_end_time - self.inference_start_time
                    if inference_phase_duration > 0:
                        logger.info(f"实际推理速度: {self.stats['total_inferences'] / inference_phase_duration:.1f} FPS")
                else:
                    logger.info("实际推理速度: 无法计算")
        else:
            logger.info("\n时间统计信息: 无法获取（计时错误）")

        logger.info("=" * 50)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8目标检测和分类脚本（支持并行处理）')
    
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
    parser.add_argument('--device', '-d', type=str, default='auto',
                       help='推理设备: auto(自动), cpu, cuda, 0, 1等（默认: auto）')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='是否保存可视化检测结果')
    parser.add_argument('--max-workers', '-w', type=int, default=4,
                       help='最大并行工作线程数（默认: 4）')
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                       help='图像批处理大小（默认: 8）')
    parser.add_argument('--debug', '-dbg', action='store_true',
                       help='启用调试模式，显示被过滤掉的低置信度检测结果')
    parser.add_argument('--save-labels', '-sl', action='store_true',
                       help='是否保存YOLO格式标签')
    parser.add_argument('--single-folder', '-sf', action='store_true',
                       help='将所有检测结果保存在单个文件夹中（不按类别分类）')
    
    args = parser.parse_args()
    
    # 根据debug参数设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("已启用调试模式")
    
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
    
    if args.max_workers < 1:
        logger.error("并行线程数必须大于0")
        return
    
    if args.batch_size < 1:
        logger.error("批处理大小必须大于0")
        return
    
    # 验证设备参数
    valid_devices = ['auto', 'cpu', 'cuda', '0', '1', '2', '3']
    if args.device not in valid_devices and not args.device.startswith('cuda:'):
        logger.error(f"无效的设备参数: {args.device}，有效值: {valid_devices} 或 cuda:N")
        return
    
    try:
        # 初始化检测器
        detector = YOLODetectionClassifier(
            model_path=args.model,
            classes_path=args.classes,
            confidence_threshold=args.confidence,
            device=args.device
        )
        
        # 初始化文件处理器
        processor = FileProcessor(
            input_path=args.input,
            output_path=args.output,
            detector=detector,
            visualize=args.visualize,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            save_labels=args.save_labels,
            single_folder=args.single_folder
        )
        
        # 处理所有文件
        processor.process_all_files()
        
    except Exception as e:
        logger.error(f"程序执行时发生错误: {e}")
        raise

if __name__ == "__main__":
    main() 