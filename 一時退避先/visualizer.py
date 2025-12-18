#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from mediapipe.tasks.python import vision
from .config import HandPoseConfig


class Visualizer:
    """
    2D/3D可視化を行うクラス（PoseとHandsの両モデル対応）
    """
    
    def __init__(self, enable_3d: bool = True):
        """
        初期化
        
        Args:
            enable_3d: 3D可視化を有効にするか
        """
        self.config = HandPoseConfig()
        self.enable_3d = enable_3d
        
        # 3D可視化用の初期化
        self.plt = None
        self.fig = None
        self.ax = None
        self.ax_hand = None  # 手専用の3Dプロット
        
        if self.enable_3d:
            self._setup_3d_visualization()
        
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
        
        print(f"Visualizer初期化完了:")
        print(f"  3D可視化: {self.enable_3d}")

    def _setup_3d_visualization(self) -> None:
        """
        3D可視化の設定（Poseと手の両方）
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            self.plt = plt
            self.fig = plt.figure(figsize=(15, 8))
            
            # Pose用の3Dプロット
            self.ax = self.fig.add_subplot(121, projection='3d')
            self.ax.set_title('Body Pose (from Pose model)')
            
            # 手用の3Dプロット
            self.ax_hand = self.fig.add_subplot(122, projection='3d')
            self.ax_hand.set_title('Right Hand (from Hands model)')
            
            self.fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)
            
            # インタラクティブモードを有効にする
            plt.ion()
            
            print("3D可視化セットアップ完了（Pose + Hands）")
            
        except ImportError as e:
            print(f"matplotlib/mpl_toolkitsのインポートエラー: {e}")
            print("3D可視化を無効にします")
            self.enable_3d = False
        except Exception as e:
            print(f"3D可視化セットアップエラー: {e}")
            self.enable_3d = False

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

    def draw_3d_landmarks_with_hands(
        self, 
        pose_result: vision.PoseLandmarkerResult,
        pose_data: Optional[Dict] = None,
        hand_world_landmarks = None
    ) -> None:
        """
        3Dワールド座標でPoseとHandsの両方を描画
        
        Args:
            pose_result: Poseの検出結果
            pose_data: 追加の姿勢データ
            hand_world_landmarks: 手のワールドランドマーク
        """
        if not self.enable_3d or self.ax is None:
            return
        
        try:
            # Poseの3D描画
            if pose_result and pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                pose_world_landmarks = pose_result.pose_world_landmarks[0]
                self._draw_3d_body_parts(pose_world_landmarks)
                
                if pose_data:
                    self._draw_3d_hand_pose(pose_data, pose_world_landmarks)
            
            # Handsの3D描画（右手専用プロット）
            if hand_world_landmarks and self.ax_hand:
                self._draw_3d_hand_detailed(hand_world_landmarks)
            
            self.plt.pause(0.001)
            
        except Exception as e:
            print(f"3D描画エラー: {e}")

    def _draw_3d_hand_detailed(self, hand_world_landmarks) -> None:
        """
        手の詳細な3D描画（Handsモデル用）
        
        Args:
            hand_world_landmarks: 手のワールドランドマーク
        """
        if not self.ax_hand:
            return
        
        try:
            # プロットをクリア
            self.ax_hand.cla()
            self.ax_hand.set_xlim3d(-0.15, 0.15)
            self.ax_hand.set_ylim3d(-0.15, 0.15)
            self.ax_hand.set_zlim3d(-0.15, 0.15)
            
            # ランドマーク座標を取得
            landmark_dict = {}
            for index, landmark in enumerate(hand_world_landmarks):
                landmark_dict[index] = [landmark.x, landmark.y, landmark.z]
            
            # 各指のリスト
            palm_list = [0, 1, 5, 9, 13, 17, 0]
            thumb_list = [1, 2, 3, 4]
            index_finger_list = [5, 6, 7, 8]
            middle_finger_list = [9, 10, 11, 12]
            ring_finger_list = [13, 14, 15, 16]
            pinky_list = [17, 18, 19, 20]
            
            # 各指を異なる色で描画
            finger_lists = [
                (palm_list, 'blue', 'Palm'),
                (thumb_list, 'red', 'Thumb'),
                (index_finger_list, 'green', 'Index'),
                (middle_finger_list, 'yellow', 'Middle'),
                (ring_finger_list, 'cyan', 'Ring'),
                (pinky_list, 'magenta', 'Pinky')
            ]
            
            for finger_indices, color, label in finger_lists:
                x_coords, y_coords, z_coords = [], [], []
                for idx in finger_indices:
                    if idx in landmark_dict:
                        point = landmark_dict[idx]
                        x_coords.append(point[0])
                        y_coords.append(point[2])  # Y-Z swap for better view
                        z_coords.append(point[1] * (-1))
                
                if x_coords:
                    self.ax_hand.plot(x_coords, y_coords, z_coords, 
                                     color=color, linewidth=2, label=label)
                    self.ax_hand.scatter(x_coords, y_coords, z_coords, 
                                        color=color, s=30, alpha=0.8)
            
            # 軸ラベルと凡例
            self.ax_hand.set_xlabel('X')
            self.ax_hand.set_ylabel('Y')
            self.ax_hand.set_zlabel('Z')
            self.ax_hand.set_title('Right Hand Detail (Hands Model)')
            self.ax_hand.legend(loc='upper right', fontsize=8)
            
        except Exception as e:
            print(f"手の詳細3D描画エラー: {e}")

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

    def _draw_3d_body_parts(self, pose_world_landmarks) -> None:
        """
        3D身体部位を描画（既存のメソッド）
        """
        try:
            landmark_dict = {}
            for index, landmark in enumerate(pose_world_landmarks):
                x = -landmark.x
                y = landmark.z
                z = landmark.y * (-1)
                landmark_dict[index] = [x, y, z]
            
            self.ax.cla()
            self.ax.set_xlim3d(*self.config.plot_limits)
            self.ax.set_ylim3d(*self.config.plot_limits)
            self.ax.set_zlim3d(*self.config.plot_limits)
            
            self._plot_body_part(landmark_dict, self.config.body_parts['face'], 'blue', marker='o')
            self._plot_body_part(landmark_dict, self.config.body_parts['left_arm'], 'green', marker='s')
            self._plot_body_part(landmark_dict, self.config.body_parts['right_arm'], 'red', marker='^')
            self._plot_body_part(landmark_dict, self.config.body_parts['left_leg'], 'cyan', marker='d')
            self._plot_body_part(landmark_dict, self.config.body_parts['right_leg'], 'magenta', marker='d')
            self._plot_body_part(landmark_dict, self.config.body_parts['torso'], 'orange', marker='*')
            
            self._draw_3d_connections(landmark_dict)
            
            self.ax.scatter([0], [0], [0], color='black', s=100, marker='o', label='Origin')
            
            self.ax.set_xlabel('X (Right-Left)')
            self.ax.set_ylabel('Y (Forward-Backward)')
            self.ax.set_zlabel('Z (Up-Down)')
            self.ax.set_title("3D Pose Visualization (Mirrored)")
            
        except Exception as e:
            print(f"3D身体部位描画エラー: {e}")
         
    def _plot_body_part(self, landmark_dict: Dict, indices: List[int], color: str, marker: str) -> None:
        """
        特定の身体部位をプロット
        
        Args:
            landmark_dict: ランドマーク辞書
            indices: 身体部位のインデックスリスト
            color: 描画色
            marker: マーカー形状
        """
        x_coords, y_coords, z_coords = [], [], []
        
        for idx in indices:
            if idx in landmark_dict:
                point = landmark_dict[idx]
                x_coords.append(point[0])
                y_coords.append(point[1])
                z_coords.append(point[2])
        
        if x_coords:  # データがある場合のみ描画
            if len(x_coords) > 1:
                # 線でつなぐ
                self.ax.plot(x_coords, y_coords, z_coords, color=color, linewidth=2)
            # 点を描画
            self.ax.scatter(x_coords, y_coords, z_coords, color=color, s=50, marker=marker, alpha=0.8)

    def _draw_3d_connections(self, landmark_dict: Dict) -> None:
        """
        3D接続線を描画
        
        Args:
            landmark_dict: ランドマーク辞書
        """
        # 主要な接続線のみ描画（見やすさのため）
        main_connections = [
            [11, 12],  # 肩
            [23, 24],  # 腰
            [11, 23],  # 左の胴体
            [12, 24],  # 右の胴体
        ]
        
        for connection in main_connections:
            if connection[0] in landmark_dict and connection[1] in landmark_dict:
                p1 = landmark_dict[connection[0]]
                p2 = landmark_dict[connection[1]]
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                           color='gray', linewidth=1, alpha=0.6)

    def _draw_3d_hand_pose(self, pose_data: Dict, pose_world_landmarks) -> None:
        """
        3D右手姿勢を描画
        
        Args:
            pose_data: 姿勢データ（位置とクォータニオン）
            pose_world_landmarks: ワールドランドマーク
        """
        try:
            from scipy.spatial.transform import Rotation as R
            
            # 手首位置
            wrist_pos = np.array(pose_data['position'])
            
            # クォータニオンから回転行列を計算
            quat = pose_data['quaternion']  # [x, y, z, w]
            rotation = R.from_quat(quat)
            rotation_matrix = rotation.as_matrix()
            
            # 座標軸を描画
            axis_length = self.config.axis_length
            colors = ['red', 'green', 'blue']
            labels = ['X-axis', 'Y-axis', 'Z-axis']
            
            for i, (color, label) in enumerate(zip(colors, labels)):
                # 基本軸ベクトル
                axis_vec = np.zeros(3)
                axis_vec[i] = axis_length
                
                # 回転を適用
                rotated_axis = rotation_matrix @ axis_vec
                
                # 手首から軸方向へのベクトルを描画
                end_pos = wrist_pos + rotated_axis
                
                self.ax.plot(
                    [wrist_pos[0], end_pos[0]],
                    [wrist_pos[1], end_pos[1]],
                    [wrist_pos[2], end_pos[2]],
                    color=color, linewidth=self.config.axis_linewidth, label=label
                )
            
            # 手首位置を強調表示
            self.ax.scatter(
                [wrist_pos[0]], [wrist_pos[1]], [wrist_pos[2]],
                color='orange', s=self.config.marker_size, 
                marker='o', label='Right Wrist', alpha=0.9
            )
            
            # 姿勢計算に使用した点を表示（デバッグ用）
            self._draw_pose_calculation_points(pose_world_landmarks)
            
            # 姿勢情報をテキスト表示
            self._draw_3d_pose_info(pose_data)
            
            # 凡例を表示
            self.ax.legend(loc='upper right', fontsize=8)
            
        except Exception as e:
            print(f"3D手姿勢描画エラー: {e}")

    def _draw_pose_calculation_points(self, pose_world_landmarks) -> None:
        """
        姿勢計算に使用した参照点を描画（ミラー表示固定）
        
        Args:
            pose_world_landmarks: ワールドランドマーク
        """
        try:
            # 右手の主要ポイント
            key_indices = [15, 17, 19]  # 手首、小指、人差し指
            colors = ['orange', 'purple', 'cyan']
            
            for idx, color in zip(key_indices, colors):
                if idx < len(pose_world_landmarks):
                    landmark = pose_world_landmarks[idx]
                    # ミラー表示固定のためX座標を反転
                    x = -landmark.x
                    y = landmark.z
                    z = landmark.y * (-1)
                    
                    self.ax.scatter([x], [y], [z], color=color, s=50, alpha=0.7)
                    
        except Exception as e:
            print(f"参照点描画エラー: {e}")

    def _draw_3d_pose_info(self, pose_data: Dict) -> None:
        """
        3D姿勢情報をテキスト表示
        
        Args:
            pose_data: 姿勢データ
        """
        try:
            pos = pose_data['position']
            quat = pose_data['quaternion']
            
            # 位置情報
            pos_text = f'Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})'
            self.ax.text2D(0.02, 0.98, pos_text,
                          transform=self.ax.transAxes, fontsize=10,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # クォータニオン情報
            quat_text = f'Quaternion: ({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})'
            self.ax.text2D(0.02, 0.88, quat_text,
                          transform=self.ax.transAxes, fontsize=8,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
        except Exception as e:
            print(f"3D姿勢情報表示エラー: {e}")

    def set_3d_view(self, elevation: float = 20, azimuth: float = 45) -> None:
        """
        3Dビューの角度を設定
        
        Args:
            elevation: 仰角
            azimuth: 方位角
        """
        if self.enable_3d and self.ax is not None:
            try:
                self.ax.view_init(elev=elevation, azim=azimuth)
            except Exception as e:
                print(f"3Dビュー設定エラー: {e}")

    def cleanup(self) -> None:
        """
        リソースのクリーンアップ
        """
        try:
            if self.enable_3d and self.plt is not None:
                self.plt.close('all')
                print("3D可視化リソースを解放しました")
        except Exception as e:
            print(f"可視化クリーンアップエラー: {e}")

    def __del__(self):
        """
        デストラクタ：リソースの自動クリーンアップ
        """
        self.cleanup()  