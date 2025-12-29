"""
视频行为分析服务 (pose3_video_service.py)
复用 pose2_detection_service 的姿态和物体检测功能
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import time

# 复用 pose2 的检测服务
from pose2_detection_service import PoseDetectionService

logger = logging.getLogger(__name__)


class VideoBehaviorAnalysisService:
    """视频行为分析服务"""
    
    def __init__(self):
        """初始化服务"""
        self.pose_service = PoseDetectionService()
        self.output_dir = Path(__file__).parent / 'output_videos'
        self.output_dir.mkdir(exist_ok=True)
    
    def analyze_video_class(
        self,
        video_path: str,
        start_time: float,
        duration: float = 300,  # 5分钟
        pose_conf_threshold: float = 0.15,
        object_conf_threshold: float = 0.25,
        looking_up_threshold: float = 0,
        looking_down_threshold: float = -2,
        output_video: bool = True,
        progress_callback=None  # 添加进度回调
    ) -> Dict[str, Any]:
        """
        全班5分钟行为分析
        
        Args:
            video_path: 视频文件路径
            start_time: 起始时间（秒）
            duration: 分析时长（秒），默认300秒=5分钟
            pose_conf_threshold: 姿态检测置信度阈值
            object_conf_threshold: 物体检测置信度阈值
            looking_up_threshold: 抬头阈值
            looking_down_threshold: 低头阈值
            output_video: 是否输出标注视频
            
        Returns:
            分析结果字典
        """
        logger.info(f"开始全班视频分析: {video_path}, 起始时间: {start_time}s, 时长: {duration}s")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频参数
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 输出视频分辨率（压缩刷50%以减小文件大小）
        output_width = orig_width // 2
        output_height = orig_height // 2
        logger.info(f"原始分辨率: {orig_width}x{orig_height}, 输出分辨率: {output_width}x{output_height}")
        
        # 跳转到起始帧
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 计算要处理的帧数
        max_frames = int(duration * fps)
        
        # 准备输出视频（使用高压缩率）
        output_path = None
        out = None
        if output_video:
            timestamp = int(time.time())
            output_filename = f'class_analysis_{timestamp}.mp4'
            output_path = self.output_dir / output_filename
            
            # 使用 MP4V 编码器，更高的压缩率
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
            
            if not out.isOpened():
                logger.warning("mp4v 编码器不可用，尝试使用 XVID")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_filename = f'class_analysis_{timestamp}.avi'
                output_path = self.output_dir / output_filename
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
        
        # 统计数据
        frame_count = 0
        behavior_stats = {
            'listening': 0,
            'using_computer': 0,
            'using_phone': 0,
            'reading_writing': 0,
            'neutral': 0
        }
        
        logger.info(f"开始处理帧，预计处理 {max_frames} 帧")
        
        # 缓存最新的行为检测结果（用于绘制标注）
        latest_behaviors = None
        
        try:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"视频读取结束，已处理 {frame_count} 帧")
                    break
                
                # 每50帧检测一次（更新标注信息）
                if frame_count % 50 == 0:
                    # 执行行为检测（不绘制，只获取结果）
                    result = self.pose_service.analyze_behavior_frame(
                        frame,
                        pose_conf_threshold=pose_conf_threshold,
                        object_conf_threshold=object_conf_threshold,
                        draw_skeleton=False,  # 不绘制，稍后手动绘制
                        draw_bbox=False,
                        looking_up_threshold=looking_up_threshold,
                        looking_down_threshold=looking_down_threshold
                    )
                    
                    # 更新统计
                    if result['success'] and result['behavior_stats']:
                        stats = result['behavior_stats']
                        for behavior, count in stats.items():
                            if behavior in behavior_stats:
                                behavior_stats[behavior] += count
                    
                    # 缓存行为检测结果
                    if result['success'] and result.get('behaviors'):
                        latest_behaviors = result['behaviors']
                
                # 在当前帧上绘制最新的标注
                if output_video and out is not None:
                    if latest_behaviors is not None:
                        # 在原始帧上绘制标注
                        annotated_frame = self.pose_service._draw_behaviors(
                            frame.copy(),
                            latest_behaviors,
                            draw_skeleton=True,
                            draw_bbox=True
                        )
                        # 缩放帧以减小文件大小
                        resized_frame = cv2.resize(annotated_frame, (output_width, output_height))
                        out.write(resized_frame)
                    else:
                        # 第一次检测前，写入原始帧（缩放后）
                        resized_frame = cv2.resize(frame, (output_width, output_height))
                        out.write(resized_frame)
                
                frame_count += 1
                
                # 更新进度
                if progress_callback and frame_count % 50 == 0:
                    progress = int((frame_count / max_frames) * 100)
                    progress_callback(progress)
                
                # 每100帧打印一次进度
                if frame_count % 100 == 0:
                    progress = int((frame_count / max_frames) * 100)
                    logger.info(f"已处理 {frame_count}/{max_frames} 帧 ({progress}%)")
        
        finally:
            cap.release()
            if out is not None:
                out.release()
            logger.info("视频处理完成，释放资源")
        
        result = {
            'success': True,
            'total_frames': frame_count,
            'duration_seconds': duration,
            'behavior_stats': behavior_stats,
            'output_video_path': str(output_path) if output_path else None
        }
        
        logger.info(f"分析完成: {result}")
        return result
    
    def analyze_video_individual(
        self,
        video_path: str,
        start_time: float,
        target_bbox: Dict[str, float],
        duration: float = 2700,  # 45分钟
        pose_conf_threshold: float = 0.15,
        object_conf_threshold: float = 0.25,
        looking_up_threshold: float = 0,
        looking_down_threshold: float = -2,
        progress_callback=None  # 添加进度回调
    ) -> Dict[str, Any]:
        """
        个人45分钟行为追踪（优化版：跳帧采样 + 区域裁剪）
        
        Args:
            video_path: 视频文件路径
            start_time: 起始时间（秒）
            target_bbox: 目标学生边界框 {x, y, w, h}
            duration: 分析时长（秒），默认2700秒=45分钟
            pose_conf_threshold: 姿态检测置信度阈值
            object_conf_threshold: 物体检测置信度阈值
            looking_up_threshold: 抬头阈值
            looking_down_threshold: 低头阈值
            
        Returns:
            行为统计结果
        """
        logger.info(f"开始个人视频追踪: {video_path}, 起始时间: {start_time}s, 时长: {duration}s")
        logger.info(f"目标边界框: {target_bbox}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频参数
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 计算要处理的总帧数
        start_frame = int(start_time * fps)
        max_frames = int(duration * fps)
        
        # 【优化1】采样间隔：10秒检测一次
        detection_interval_seconds = 10
        detection_interval_frames = int(fps * detection_interval_seconds)
        
        # 计算需要检测的帧索引列表
        frame_indices = list(range(0, max_frames, detection_interval_frames))
        total_samples = len(frame_indices)
        
        logger.info(f"采样策略: 每{detection_interval_seconds}秒检测一次 (间隔{detection_interval_frames}帧)")
        logger.info(f"总帧数: {max_frames}, 需要采样: {total_samples} 次")
        
        # 【优化2】目标区域坐标（裁剪用）
        target_x = int(target_bbox['x'])
        target_y = int(target_bbox['y'])
        target_w = int(target_bbox['w'])
        target_h = int(target_bbox['h'])
        
        crop_x1 = max(0, target_x)
        crop_y1 = max(0, target_y)
        
        # 统计数据
        sampled_count = 0  # 成功采样次数
        behavior_stats = {
            'listening': 0,
            'using_computer': 0,
            'using_phone': 0,
            'reading_writing': 0,
            'neutral': 0
        }
        
        logger.info(f"开始处理，裁剪区域: ({crop_x1}, {crop_y1}), 大小: {target_w}x{target_h}")
        
        try:
            # 【优化核心】遍历采样帧索引，直接跳转
            for i, frame_offset in enumerate(frame_indices):
                # 计算绝对帧位置
                absolute_frame_index = start_frame + frame_offset
                
                # 【优化：直接跳转到目标帧，不读取中间帧】
                cap.set(cv2.CAP_PROP_POS_FRAMES, absolute_frame_index)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"第{i+1}次采样失败：无法读取帧 {absolute_frame_index}")
                    continue
                
                # 更新进度（每10次采样更新一次，减少HTTP请求）
                if i % 10 == 0 and progress_callback:
                    current_progress = min(99, int((i / total_samples) * 100))
                    progress_callback(current_progress)
                
                # 每1000次采样输出一次日志
                if i % 1000 == 0:
                    logger.info(f"采样进度: {i}/{total_samples} ({i*100//total_samples}%)")
                
                # 【优化：裁剪目标区域，只检测该区域】
                crop_x2 = min(frame.shape[1], target_x + target_w)
                crop_y2 = min(frame.shape[0], target_y + target_h)
                cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # 检查裁剪后的帧是否有效
                if cropped_frame.size == 0:
                    logger.warning(f"第{i+1}次采样：裁剪区域无效")
                    continue
                
                # 执行行为检测（只检测裁剪后的小区域，速度更快）
                result = self.pose_service.analyze_behavior_frame(
                    cropped_frame,
                    pose_conf_threshold=pose_conf_threshold,
                    object_conf_threshold=object_conf_threshold,
                    draw_skeleton=False,
                    draw_bbox=False,
                    looking_up_threshold=looking_up_threshold,
                    looking_down_threshold=looking_down_threshold
                )
                
                # 统计行为（取第一个检测到的人）
                if result['success'] and result.get('behaviors'):
                    if len(result['behaviors']) > 0:
                        target_behavior = result['behaviors'][0]
                        behavior_type = target_behavior['behavior']
                        if behavior_type in behavior_stats:
                            behavior_stats[behavior_type] += 1
                            sampled_count += 1
                
        finally:
            cap.release()
            logger.info("视频处理完成，释放资源")
        
        # 构建结果
        result = {
            'success': True,
            'total_frames': max_frames,
            'duration_seconds': duration,
            'behavior_stats': behavior_stats,
            'sampled_frames': sampled_count,  # 实际成功采样次数
            'total_samples': total_samples     # 计划采样次数
        }
        
        # 计算行为时长（分钟）
        behavior_minutes = {}
        for behavior, count in behavior_stats.items():
            if sampled_count > 0:
                # 基于采样比例推算总时长
                duration_minutes = (count / sampled_count) * (duration / 60)
                behavior_minutes[f'{behavior}_minutes'] = round(duration_minutes, 2)
            else:
                behavior_minutes[f'{behavior}_minutes'] = 0.0
        
        result['behavior_minutes'] = behavior_minutes
        
        logger.info(f"\n" + "="*50)
        logger.info(f"分析完成！")
        logger.info(f"  总帧数: {max_frames}")
        logger.info(f"  计划采样: {total_samples} 次")
        logger.info(f"  成功采样: {sampled_count} 次")
        logger.info(f"  采样成功率: {sampled_count*100//total_samples if total_samples > 0 else 0}%")
        logger.info(f"  行为统计: {behavior_stats}")
        logger.info(f"  行为时长(分钟): {behavior_minutes}")
        logger.info("="*50 + "\n")
        
        return result
    
    def _find_target_person(
        self,
        behaviors: list,
        target_x: int,
        target_y: int,
        target_w: int,
        target_h: int
    ) -> Optional[Dict]:
        """
        在检测结果中查找目标学生
        
        Args:
            behaviors: 行为检测结果列表
            target_x, target_y, target_w, target_h: 目标边界框
            
        Returns:
            匹配的行为数据，如果没有匹配则返回None
        """
        target_center_x = target_x + target_w / 2
        target_center_y = target_y + target_h / 2
        
        best_match = None
        min_distance = float('inf')
        
        for behavior in behaviors:
            bbox = behavior['bbox']
            person_center_x = (bbox['x1'] + bbox['x2']) / 2
            person_center_y = (bbox['y1'] + bbox['y2']) / 2
            
            # 计算中心点距离
            distance = np.sqrt(
                (person_center_x - target_center_x) ** 2 +
                (person_center_y - target_center_y) ** 2
            )
            
            # 找最近的人物
            if distance < min_distance:
                min_distance = distance
                best_match = behavior
        
        # 如果距离太远（超过目标框宽度的2倍），认为没有匹配
        if min_distance > target_w * 2:
            return None
        
        return best_match


# 全局服务实例
_video_analysis_service = None


def get_video_analysis_service() -> VideoBehaviorAnalysisService:
    """获取视频分析服务单例"""
    global _video_analysis_service
    if _video_analysis_service is None:
        _video_analysis_service = VideoBehaviorAnalysisService()
    return _video_analysis_service


if __name__ == "__main__":
    # 简单测试
    service = VideoBehaviorAnalysisService()
    print("视频行为分析服务初始化成功")
