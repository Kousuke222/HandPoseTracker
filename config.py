#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Union
import os


class HandPoseConfig:
    """
    MediaPipe Hand Pose検出の設定クラス
    """
    
    def __init__(self):
        self.setup_model_config()
        self.setup_landmark_config()
        self.setup_hand_orientation_config()
        self.setup_visualization_config()
        self.setup_ros_config()

    def setup_model_config(self):
        """MediaPipeモデル関連の設定"""
        # モデルURL（heavyモデル固定）
        self.model_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task'

        # モデル保存ディレクトリ
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(script_dir, '..', 'models')

        # 検出信頼度のしきい値
        self.min_detection_confidence = 0.5
        self.min_tracking_confidence = 0.5

    def setup_landmark_config(self):
        """ランドマーク関連の設定"""
        # ランドマーク描画情報
        self.landmark_draw_info: Dict[int, Dict[str, Union[str, Tuple[int, int, int]]]] = {
            0: {'name': 'NOSE', 'color': (0, 255, 0)},
            1: {'name': 'LEFT_EYE_INNER', 'color': (255, 0, 0)},
            2: {'name': 'LEFT_EYE', 'color': (0, 0, 255)},
            3: {'name': 'LEFT_EYE_OUTER', 'color': (255, 255, 0)},
            4: {'name': 'RIGHT_EYE_INNER', 'color': (0, 255, 255)},
            5: {'name': 'RIGHT_EYE', 'color': (255, 0, 255)},
            6: {'name': 'RIGHT_EYE_OUTER', 'color': (128, 128, 128)},
            7: {'name': 'LEFT_EAR', 'color': (255, 128, 0)},
            8: {'name': 'RIGHT_EAR', 'color': (128, 0, 255)},
            9: {'name': 'MOUTH_LEFT', 'color': (0, 128, 255)},
            10: {'name': 'MOUTH_RIGHT', 'color': (128, 255, 0)},
            11: {'name': 'LEFT_SHOULDER', 'color': (255, 128, 128)},
            12: {'name': 'RIGHT_SHOULDER', 'color': (128, 128, 0)},
            13: {'name': 'LEFT_ELBOW', 'color': (0, 128, 128)},
            14: {'name': 'RIGHT_ELBOW', 'color': (128, 0, 128)},
            15: {'name': 'LEFT_WRIST', 'color': (64, 64, 64)},
            16: {'name': 'RIGHT_WRIST', 'color': (192, 192, 192)},
            17: {'name': 'LEFT_PINKY', 'color': (255, 69, 0)},
            18: {'name': 'RIGHT_PINKY', 'color': (75, 0, 130)},
            19: {'name': 'LEFT_INDEX', 'color': (173, 255, 47)},
            20: {'name': 'RIGHT_INDEX', 'color': (220, 20, 60)},
            21: {'name': 'LEFT_THUMB', 'color': (255, 0, 0)},
            22: {'name': 'RIGHT_THUMB', 'color': (0, 0, 255)},
            23: {'name': 'LEFT_HIP', 'color': (0, 255, 0)},
            24: {'name': 'RIGHT_HIP', 'color': (255, 255, 0)},
            25: {'name': 'LEFT_KNEE', 'color': (0, 255, 255)},
            26: {'name': 'RIGHT_KNEE', 'color': (255, 0, 255)},
            27: {'name': 'LEFT_ANKLE', 'color': (128, 128, 128)},
            28: {'name': 'RIGHT_ANKLE', 'color': (255, 128, 0)},
            29: {'name': 'LEFT_HEEL', 'color': (128, 0, 255)},
            30: {'name': 'RIGHT_HEEL', 'color': (0, 128, 255)},
            31: {'name': 'LEFT_FOOT_INDEX', 'color': (128, 255, 0)},
            32: {'name': 'RIGHT_FOOT_INDEX', 'color': (255, 128, 128)}
        }
        
        # 接続線の定義
        self.connection_lines: List[List[int]] = [
            [0, 1], [1, 2], [2, 3], [3, 7],      # 顔左側
            [0, 4], [4, 5], [5, 6], [6, 8],      # 顔右側
            [9, 10],                              # 口
            [11, 12],                             # 肩
            [11, 13], [13, 15],                   # 左腕
            [15, 17], [15, 19], [15, 21],         # 左手
            [12, 14], [14, 16],                   # 右腕
            [16, 18], [16, 20], [16, 22],         # 右手
            [23, 24],                             # 腰
            [23, 25], [25, 27], [27, 29], [29, 31], # 左脚
            [24, 26], [26, 28], [28, 30], [30, 32], # 右脚
            [11, 23], [12, 24]                    # 胴体
        ]
        
        # 右腕のランドマークインデックス
        self.right_arm_indices = [11, 13, 15, 17, 19, 21]  # [肩, 肘, 手首, 小指, 人差し指, 親指]
        
        # 身体部位別のインデックス
        self.body_parts = {
            'face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'left_arm': [12, 14, 16, 18, 20, 22],
            'right_arm': [11, 13, 15, 17, 19, 21],
            'left_leg': [23, 25, 27, 29, 31],
            'right_leg': [24, 26, 28, 30, 32],
            'torso': [11, 12, 23, 24]
        }

    def setup_hand_orientation_config(self):
        """手のひら向き計算関連の設定"""
        # 最小二乗平面法による向き計算の有効/無効（デフォルト: 有効）
        self.use_hands_orientation = True

        # Handsモデルのランドマークインデックス
        self.hand_mcp_indices = {
            'thumb': 1,      # 親指の付け根（CMC）
            'index': 5,      # 人差し指の付け根（MCP）
            'middle': 9,     # 中指の付け根（MCP）
            'ring': 13,      # 薬指の付け根（MCP）
            'pinky': 17      # 小指の付け根（MCP）
        }
        self.hand_wrist_index = 0
        self.hand_middle_tip_index = 12

        # 最小二乗平面法関連
        self.min_mcp_landmarks = 3  # 最低必要なMCPランドマーク数
        self.planarity_threshold = 0.01  # 特異値比の閾値
        self.orientation_timeout = 3.0  # 秒（検出タイムアウト）

        # クォータニオン平滑化
        self.use_slerp_smoothing = True  # SLERP補間の使用
        self.slerp_alpha = 0.3  # SLERP補間の重み（0-1）

    def setup_visualization_config(self):
        """可視化関連の設定"""
        # 2D描画設定
        self.circle_radius = 5
        self.line_thickness = 3
        self.font_scale = 1.0
        self.font_thickness = 2
        self.line_color = (220, 220, 220)

        # FPS表示設定
        self.fps_position = (10, 30)
        self.fps_color_normal = (0, 255, 0)

        # ウィンドウ設定
        self.window_name = 'MediaPipe Hand Pose Detection'
        self.window_width = 960
        self.window_height = 540

    def setup_ros_config(self):
        """ROS関連の設定"""
        # トピック名
        self.pose_topic = 'target_pose'
        self.hand_control_topic = 'hand_control'

        # QoS設定
        self.queue_size = 10

        # パブリッシュレート
        self.default_publish_rate = 60.0  # Hz

        # ノード名
        self.node_name = 'mp_hand_pose_publisher'

    def get_model_path(self) -> str:
        """
        モデルファイルのパスを取得（heavyモデル固定）

        Returns:
            モデルファイルの完全パス
        """
        model_name = self.model_url.split('/')[-1]
        quantize_type = self.model_url.split('/')[-3]
        split_name = model_name.split('.')
        model_filename = f"{split_name[0]}_{quantize_type}.{split_name[1]}"

        return os.path.join(self.model_dir, model_filename)

    def validate_camera_params(self, device: int, width: int, height: int) -> Tuple[int, int, int]:
        """
        カメラパラメータの検証と補正
        
        Args:
            device: カメラデバイス番号
            width: 幅
            height: 高さ
            
        Returns:
            検証済みの(device, width, height)
        """
        # デバイス番号の検証
        if device < 0:
            device = 0
            
        # 解像度の検証
        if width <= 0 or width > 4096:
            width = 960
        if height <= 0 or height > 2160:
            height = 540
            
        return device, width, height

    def get_landmark_color(self, landmark_index: int) -> Tuple[int, int, int]:
        """
        ランドマークの描画色を取得
        
        Args:
            landmark_index: ランドマークのインデックス
            
        Returns:
            BGR色タプル
        """
        if landmark_index in self.landmark_draw_info:
            return self.landmark_draw_info[landmark_index]['color']
        else:
            return (128, 128, 128)  # デフォルトグレー

    def get_landmark_name(self, landmark_index: int) -> str:
        """
        ランドマークの名前を取得
        
        Args:
            landmark_index: ランドマークのインデックス
            
        Returns:
            ランドマーク名
        """
        if landmark_index in self.landmark_draw_info:
            return self.landmark_draw_info[landmark_index]['name']
        else:
            return f'UNKNOWN_{landmark_index}'