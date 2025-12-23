#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from mediapipe.tasks.python import vision
from .config import HandPoseConfig


class Visualizer:
    """
    2D可視化を行うクラス（PoseとHandsの両モデル対応）
    """

    def __init__(self):
        """
        初期化
        """
        self.config = HandPoseConfig()

        # 手のランドマーク描画情報
        self.hand_landmark_info = {
            0: {'name': 'WRIST', 'color': (0, 255, 0)},
            1: {'name': 'THUMB_CMC', 'color': (255, 0, 0)},
            2: {'name': 'THUMB_MCP', 'color': (0, 0, 255)},
            3: {'name': 'THUMB_IP', 'color': (255, 255, 0)},
            4: {'name': 'THUMB_TIP', 'color': (0, 255, 255)},
            5: {'name': 'INDEX_FINGER_MCP', 'color': (255, 0, 255)},
            6: {'name': 'INDEX_FINGER_PIP', 'color': (128, 128, 128)},
            7: {'name': 'INDEX_FINGER_DIP', 'color': (255, 128, 0)},
            8: {'name': 'INDEX_FINGER_TIP', 'color': (128, 0, 255)},
            9: {'name': 'MIDDLE_FINGER_MCP', 'color': (0, 128, 255)},
            10: {'name': 'MIDDLE_FINGER_PIP', 'color': (128, 255, 0)},
            11: {'name': 'MIDDLE_FINGER_DIP', 'color': (255, 128, 128)},
            12: {'name': 'MIDDLE_FINGER_TIP', 'color': (128, 128, 0)},
            13: {'name': 'RING_FINGER_MCP', 'color': (0, 128, 128)},
            14: {'name': 'RING_FINGER_PIP', 'color': (128, 0, 128)},
            15: {'name': 'RING_FINGER_DIP', 'color': (64, 64, 64)},
            16: {'name': 'RING_FINGER_TIP', 'color': (192, 192, 192)},
            17: {'name': 'PINKY_MCP', 'color': (255, 69, 0)},
            18: {'name': 'PINKY_PIP', 'color': (75, 0, 130)},
            19: {'name': 'PINKY_DIP', 'color': (173, 255, 47)},
            20: {'name': 'PINKY_TIP', 'color': (220, 20, 60)}
        }
        
        # 手の接続線情報
        self.hand_connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],    # 親指
            [0, 5], [5, 6], [6, 7], [7, 8],    # 人差し指
            [0, 9], [9, 10], [10, 11], [11, 12],  # 中指
            [0, 13], [13, 14], [14, 15], [15, 16],  # 薬指
            [0, 17], [17, 18], [18, 19], [19, 20]   # 小指
        ]

        print(f"Visualizer初期化完了（2Dのみ）")

    def draw_2d_landmarks_with_hands(
        self, 
        image: np.ndarray, 
        pose_result: Optional[vision.PoseLandmarkerResult],
        hand_result: Optional[Dict],
        fps: float = 0.0,
        hand_status: str = "Unknown",
        hand_confidence: float = 0.0
    ) -> np.ndarray:
        """
        2D画像上にPoseとHandsの両方のランドマークを描画
        
        Args:
            image: 描画対象の画像
            pose_result: Poseの検出結果
            hand_result: Handsの検出結果
            fps: FPS値
            hand_status: 手の開閉状態
            hand_confidence: 状態判定の信頼度
            
        Returns:
            描画済みの画像
        """
        if pose_result is None and hand_result is None:
            return self._draw_fps_and_status_only(image, fps, hand_status, hand_confidence)
        
        try:
            image_height, image_width = image.shape[:2]
            
            # Poseランドマークの描画
            if pose_result and pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                pose_landmarks = pose_result.pose_landmarks[0]
                
                # ランドマーク情報を整理
                landmark_dict = self._create_2d_landmark_dict(
                    pose_landmarks, image_width, image_height
                )
                
                # 接続線を描画
                self._draw_connection_lines(image, landmark_dict)
                
                # 各ランドマークを描画
                self._draw_landmark_points(image, landmark_dict)
            
            # Handsランドマークの描画（右手のみ）
            if hand_result and hand_result['right_hand']['landmarks']:
                self._draw_hand_landmarks_2d(
                    image, 
                    hand_result['right_hand']['landmarks'],
                    image_width,
                    image_height,
                    hand_status,
                    hand_confidence
                )
            
            # 左手のラベル表示（検出されている場合）
            if hand_result and hand_result['left_hand']['landmarks']:
                self._draw_left_hand_label(
                    image,
                    hand_result['left_hand']['landmarks'],
                    image_width,
                    image_height
                )
            
            # FPSと手の状態表示
            self._draw_fps_and_hand_status(image, fps, hand_status, hand_confidence)
            
            return image
            
        except Exception as e:
            print(f"2D描画エラー: {e}")
            return image

    def _draw_hand_landmarks_2d(
        self,
        image: np.ndarray,
        hand_landmarks,
        image_width: int,
        image_height: int,
        hand_status: str,
        confidence: float
    ) -> None:
        """
        手のランドマークを2D描画（詳細版）
        
        Args:
            image: 描画対象画像
            hand_landmarks: 手のランドマーク
            image_width: 画像幅
            image_height: 画像高さ
            hand_status: 手の開閉状態
            confidence: 信頼度
        """
        if hand_landmarks is None:
            return
        
        # ランドマーク座標を取得
        landmark_dict = {}
        for index, landmark in enumerate(hand_landmarks):
            x = min(int(landmark.x * image_width), image_width - 1)
            y = min(int(landmark.y * image_height), image_height - 1)
            landmark_dict[index] = [x, y, landmark.z]
        
        # 接続線を描画
        for connection in self.hand_connections:
            if connection[0] in landmark_dict and connection[1] in landmark_dict:
                pt1 = tuple(landmark_dict[connection[0]][:2])
                pt2 = tuple(landmark_dict[connection[1]][:2])
                cv2.line(image, pt1, pt2, (220, 220, 220), 2, cv2.LINE_AA)
        
        # 各ランドマークを描画
        for index, coords in landmark_dict.items():
            if index in self.hand_landmark_info:
                color = self.hand_landmark_info[index]['color']
            else:
                color = (128, 128, 128)
            
            cv2.circle(image, (coords[0], coords[1]), 4, color, -1, cv2.LINE_AA)
        
        # 手首位置に状態ラベルを表示
        if 0 in landmark_dict:  # 手首
            wrist_pos = landmark_dict[0]
            
            # 状態に応じた色
            if hand_status == 'O':
                status_color = (0, 255, 0)  # 緑
                status_text = "OPEN"
            elif hand_status == 'C':
                status_color = (0, 0, 255)  # 赤
                status_text = "CLOSED"
            else:
                status_color = (128, 128, 128)  # グレー
                status_text = "UNKNOWN"
            
            # 外接矩形を計算
            x_coords = [pt[0] for pt in landmark_dict.values()]
            y_coords = [pt[1] for pt in landmark_dict.values()]
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            
            # 外接矩形を描画
            cv2.rectangle(image, (bbox[0]-5, bbox[1]-5), (bbox[2]+5, bbox[3]+5), status_color, 2)
            
            # ラベル表示
            label_text = f"RIGHT HAND: {status_text}"
            cv2.putText(
                image, label_text, (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                status_color, 2, cv2.LINE_AA
            )
            
            # 信頼度表示
            if hand_status != "Unknown":
                conf_text = f"Conf: {confidence:.2f}"
                cv2.putText(
                    image, conf_text, (bbox[0], bbox[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    status_color, 1, cv2.LINE_AA
                )

    def _draw_left_hand_label(
        self,
        image: np.ndarray,
        hand_landmarks,
        image_width: int,
        image_height: int
    ) -> None:
        """
        左手のラベルのみ表示
        
        Args:
            image: 描画対象画像
            hand_landmarks: 手のランドマーク
            image_width: 画像幅
            image_height: 画像高さ
        """
        if hand_landmarks is None:
            return
        
        # 外接矩形を計算
        x_coords = []
        y_coords = []
        for landmark in hand_landmarks:
            x = min(int(landmark.x * image_width), image_width - 1)
            y = min(int(landmark.y * image_height), image_height - 1)
            x_coords.append(x)
            y_coords.append(y)
        
        if x_coords and y_coords:
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            
            # 外接矩形を描画
            cv2.rectangle(image, (bbox[0]-5, bbox[1]-5), (bbox[2]+5, bbox[3]+5), (255, 255, 255), 2)
            
            # ラベル表示
            cv2.putText(
                image, "LEFT HAND", (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA
            )


    # 既存のメソッドはそのまま維持
    def _create_2d_landmark_dict(
        self, 
        pose_landmarks, 
        image_width: int, 
        image_height: int
    ) -> Dict[int, List[Union[int, float]]]:
        """
        2D用のランドマーク辞書を作成
        """
        landmark_dict = {}
        
        for index, landmark in enumerate(pose_landmarks):
            if hasattr(landmark, 'visibility') and landmark.visibility < 0:
                continue
            if hasattr(landmark, 'presence') and landmark.presence < 0:
                continue
            
            x = min(int(landmark.x * image_width), image_width - 1)
            y = min(int(landmark.y * image_height), image_height - 1)
            z = landmark.z if hasattr(landmark, 'z') else 0.0
            
            landmark_dict[index] = [x, y, z]
        
        return landmark_dict

    def _draw_connection_lines(self, image: np.ndarray, landmark_dict: Dict[int, List[Union[int, float]]]) -> None:
        """
        ランドマーク間の接続線を描画
        """
        for line_info in self.config.connection_lines:
            if line_info[0] in landmark_dict and line_info[1] in landmark_dict:
                pt1 = tuple(landmark_dict[line_info[0]][:2])
                pt2 = tuple(landmark_dict[line_info[1]][:2])
                
                cv2.line(
                    image, pt1, pt2, 
                    self.config.line_color,
                    self.config.line_thickness, 
                    cv2.LINE_AA
                )

    def _draw_landmark_points(self, image: np.ndarray, landmark_dict: Dict[int, List[Union[int, float]]]) -> None:
        """
        各ランドマークポイントを描画
        """
        for index, landmark in landmark_dict.items():
            color = self.config.get_landmark_color(index)
            center = (landmark[0], landmark[1])
            
            cv2.circle(
                image, center, 
                self.config.circle_radius,
                color, -1, cv2.LINE_AA
            )

    def _draw_fps_and_hand_status(
        self, 
        image: np.ndarray, 
        fps: float,
        hand_status: str,
        confidence: float
    ) -> None:
        """
        FPS情報と手の開閉状態を描画
        """
        # FPS表示
        cv2.putText(
            image, f"FPS: {fps:.1f}",
            self.config.fps_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            self.config.fps_color_normal,
            self.config.font_thickness,
            cv2.LINE_AA
        )
        
        # 手の開閉状態表示
        if hand_status == 'O':
            status_color = (0, 255, 0)  # 緑
            status_text = "OPEN"
        elif hand_status == 'C':
            status_color = (0, 0, 255)  # 赤
            status_text = "CLOSED"
        else:
            status_color = (128, 128, 128)  # グレー
            status_text = "UNKNOWN"
        
        status_position = (10, 70)
        cv2.putText(
            image, f"Hand Status: {status_text} ({confidence:.2f})",
            status_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA
        )

    def _draw_fps_and_status_only(
        self, 
        image: np.ndarray, 
        fps: float,
        hand_status: str,
        confidence: float
    ) -> np.ndarray:
        """
        検出結果がない場合のFPSと状態表示のみ
        """
        self._draw_fps_and_hand_status(image, fps, hand_status, confidence)
        
        # "NO DETECTION"メッセージ
        text = "NO POSE/HAND DETECTED"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = (image.shape[0] + text_size[1]) // 2
        
        cv2.putText(
            image, text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 2, cv2.LINE_AA
        )
        
        return image


    def cleanup(self) -> None:
        """
        リソースのクリーンアップ
        """
        pass  # 2Dのみなのでクリーンアップ不要

    def __del__(self):
        """
        デストラクタ：リソースの自動クリーンアップ
        """
        self.cleanup()  