#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple, Optional
from geometry_msgs.msg import Pose
from .config import HandPoseConfig


class PoseCalculator:
    """
    MediaPipeランドマークから右手の姿勢を計算するクラス
    """

    def __init__(self, hand_open_threshold: float = 0.4, fixed_orientation_planning: bool = True):
        """
        初期化
        
        Args:
            hand_open_threshold: 手の開閉判定閾値（0-1、低いほど開いた状態）
        """
        self.config = HandPoseConfig()
        
        # 右腕のランドマークインデックス [肩, 肘, 手首, 小指, 人差し指, 親指]
        self.right_arm_indices = [11, 13, 15, 17, 19, 21]
        
        # 手のランドマークインデックス（MediaPipe Poseモデル用）
        # Poseモデルでは手の詳細なランドマークは限定的
        self.hand_indices = {
            'wrist': 15,      # 手首
            'pinky': 17,      # 小指
            'index': 19,      # 人差し指
            'thumb': 21       # 親指
        }
        
        # 最小信頼度しきい値
        self.min_visibility = 0.5
        self.min_presence = 0.5
        
        # 平滑化用のバッファ
        self.pose_history = []
        self.history_size = 5
        
        # 手の開閉判定用
        self.hand_open_threshold = hand_open_threshold
        self.hand_status_history = []
        self.hand_status_history_size = 3
        
        self.fixed_orientation_planning = fixed_orientation_planning

    def calculate_and_convert_pose(self, pose_world_landmarks) -> Optional[Pose]:
        """
        MediaPipeのワールドランドマークからROS Poseメッセージを生成
        ミラー表示で固定
        
        Args:
            pose_world_landmarks: MediaPipeのワールドランドマーク
            
        Returns:
            ROS Poseメッセージ（計算失敗時はNone）
        """
        try:
            # ランドマークの信頼度チェック
            if not self._validate_landmarks(pose_world_landmarks):
                return None
            
            # ランドマーク辞書を作成（ミラー表示固定）
            landmark_dict = self._create_landmark_dict(pose_world_landmarks)
            
            # 右手の姿勢を計算
            rotation_matrix, quaternion, wrist_position = self._calculate_hand_pose(landmark_dict)
            
            if rotation_matrix is None:
                return None
            
            # ROS Poseメッセージに変換
            pose_msg = self._create_pose_message(wrist_position, quaternion)
            
            # 平滑化
            pose_msg = self._smooth_pose(pose_msg)
            
            return pose_msg
            
        except Exception as e:
            print(f"姿勢計算エラー: {e}")
            return None

    def calculate_hand_status(self, pose_world_landmarks) -> Tuple[str, float]:
        """
        右手の開閉状態を判定（Poseモデル用の簡易版）
        
        Args:
            pose_world_landmarks: MediaPipeのワールドランドマーク
            
        Returns:
            Tuple[str, float]: (状態('O'/'C'/'Unknown'), 信頼度(0-1))
        """
        try:
            if not pose_world_landmarks or len(pose_world_landmarks) < 22:
                return "Unknown", 0.0
            
            # ランドマーク辞書を作成
            landmark_dict = self._create_landmark_dict(pose_world_landmarks)
            
            # 必要なランドマークが存在するかチェック
            required_indices = [self.hand_indices['wrist'], 
                              self.hand_indices['pinky'],
                              self.hand_indices['index'], 
                              self.hand_indices['thumb']]
            
            for idx in required_indices:
                if idx not in landmark_dict:
                    return "Unknown", 0.0
            
            # 手の開き具合を推定（手首から各指先までの距離ベース）
            wrist = np.array(landmark_dict[self.hand_indices['wrist']])
            pinky = np.array(landmark_dict[self.hand_indices['pinky']])
            index = np.array(landmark_dict[self.hand_indices['index']])
            thumb = np.array(landmark_dict[self.hand_indices['thumb']])
            
            # 手首から各指先までの距離
            dist_pinky = np.linalg.norm(pinky - wrist)
            dist_index = np.linalg.norm(index - wrist)
            dist_thumb = np.linalg.norm(thumb - wrist)
            
            # 指同士の距離（開いているとき距離が大きくなる）
            dist_pinky_index = np.linalg.norm(pinky - index)
            dist_index_thumb = np.linalg.norm(index - thumb)
            
            # 手のサイズを推定（正規化用）
            hand_size = (dist_pinky + dist_index + dist_thumb) / 3.0
            
            if hand_size < 1e-6:
                return "Unknown", 0.0
            
            # 開き具合スコア（正規化済み）
            # 指同士が離れているほどスコアが高い
            spread_score = (dist_pinky_index + dist_index_thumb) / (2.0 * hand_size)
            
            # 手首から指先までの平均距離（正規化済み）
            extension_score = (dist_pinky + dist_index) / (2.0 * hand_size)
            
            # 総合スコア（0: 閉じた状態, 1: 開いた状態）
            total_score = (spread_score * 0.6 + extension_score * 0.4)
            
            # 信頼度の計算（スコアが閾値から離れているほど信頼度が高い）
            confidence = min(1.0, abs(total_score - self.hand_open_threshold) * 2.0)
            
            # 閾値による判定
            status = "O" if total_score > self.hand_open_threshold else "C"
            
            # 履歴による平滑化
            self.hand_status_history.append((status, confidence))
            if len(self.hand_status_history) > self.hand_status_history_size:
                self.hand_status_history.pop(0)
            
            # 履歴から最も頻出する状態を選択
            if len(self.hand_status_history) >= 2:
                status_counts = {}
                total_confidence = 0.0
                for s, c in self.hand_status_history:
                    status_counts[s] = status_counts.get(s, 0) + 1
                    total_confidence += c
                
                # 最頻値を選択
                status = max(status_counts, key=status_counts.get)
                confidence = total_confidence / len(self.hand_status_history)
            
            return status, confidence
            
        except Exception as e:
            print(f"手の状態計算エラー: {e}")
            return "Unknown", 0.0

    def _validate_landmarks(self, pose_world_landmarks) -> bool:
        """
        ランドマークの信頼度を検証
        
        Args:
            pose_world_landmarks: MediaPipeのワールドランドマーク
            
        Returns:
            検証結果（True: 有効, False: 無効）
        """
        try:
            # 右手関連のランドマークの信頼度をチェック
            for index in self.right_arm_indices:
                if index >= len(pose_world_landmarks):
                    return False
                    
                landmark = pose_world_landmarks[index]
                if (hasattr(landmark, 'visibility') and landmark.visibility < self.min_visibility) or \
                   (hasattr(landmark, 'presence') and landmark.presence < self.min_presence):
                    return False
            
            return True
            
        except Exception:
            return False

    def _create_landmark_dict(self, pose_world_landmarks) -> Dict[int, List[float]]:
        """
        ランドマークリストから辞書を作成（ミラー表示固定）
        
        Args:
            pose_world_landmarks: MediaPipeのワールドランドマーク
            
        Returns:
            ランドマーク辞書 {インデックス: [x, y, z]}
        """
        landmark_dict = {}
        for index, landmark in enumerate(pose_world_landmarks):
            # ミラー表示固定のためX座標を反転
            x = -landmark.x
            y = landmark.y
            z = landmark.z
            
            landmark_dict[index] = [x, y, z]
        return landmark_dict

    def _calculate_hand_pose(self, landmark_dict: Dict[int, List[float]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        右手の姿勢を計算する核となる関数
        
        Args:
            landmark_dict: ランドマーク辞書
            
        Returns:
            (回転行列, クォータニオン, 手首座標) のタプル
        """
        try:
            # 右腕の座標を取得（MediaPipe座標系をロボット座標系に変換）
            right_arm_points = []
            for index in self.right_arm_indices:
                if index not in landmark_dict:
                    return None, None, None
                    
                point = landmark_dict[index]
                # MediaPipe座標系 -> ロボット座標系変換
                # MediaPipe: Y軸下向き, Z軸カメラ向き
                # ロボット: X軸前向き, Y軸左向き, Z軸上向き
                converted_point = [
                    point[2] * (-1),    # X (MediaPipeのZ反転 -> ロボットのX、奥行き→前後)
                    point[0],           # Y (MediaPipeのX -> ロボットのY、左右、ミラー処理済み)
                    point[1] * (-1)     # Z (MediaPipeのY反転 -> ロボットのZ、下向き→上向き)
                ]
                right_arm_points.append(converted_point)
            
            # インデックス: [肩, 肘, 手首, 小指, 人差し指, 親指]
            #               [0,  1,  2,   3,   4,      5]
            wrist_pos = np.array(right_arm_points[2])      # 手首
            pinky_pos = np.array(right_arm_points[3])      # 小指
            index_pos = np.array(right_arm_points[4])      # 人差し指

            if not self.fixed_orientation_planning:
                # 小指と人差し指の中点を計算
                finger_midpoint = (pinky_pos + index_pos) / 2
            
                # 姿勢計算用の基準点を設定
                pts = np.array([
                    wrist_pos,          # A (手首)
                    index_pos,          # B (人差し指)
                    pinky_pos,          # C (小指)
                    finger_midpoint     # D (中点)
                ])

                # 手の座標系を定義
                # X軸: 指先方向 (手首から指の中点へ)
                x_axis = pts[3] - pts[0]
                if np.linalg.norm(x_axis) < 1e-6:
                    return None, None, None
                x_axis /= np.linalg.norm(x_axis)

                # Z軸: 手の上方向 (手のひらの法線方向)
                AB = pts[1] - pts[0]  # 手首から人差し指
                AC = pts[2] - pts[0]  # 手首から小指

                # ミラー表示固定のため外積の方向を調整
                z_axis = np.cross(AB, AC)  # ミラー時は順序を逆に

                if np.linalg.norm(z_axis) < 1e-6:
                    return None, None, None
                z_axis /= np.linalg.norm(z_axis)

                # Y軸: 手の左方向
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)

                # 回転行列を構築
                rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)

                # クォータニオンに変換
                rot = R.from_matrix(rotation_matrix)
                quaternion = rot.as_quat()  # [x, y, z, w]形式

                return rotation_matrix, quaternion, wrist_pos
            
            else:
                # 固定姿勢の場合は手首の位置とクォータニオン(初期値)を返す
                quaternion = np.array([1.0, 0.0, 0.0, 0.0])
                rotation_matrix = R.from_quat(quaternion).as_matrix()
                return rotation_matrix, quaternion, wrist_pos

        except Exception as e:
            print(f"姿勢計算の内部エラー: {e}")
            return None, None, None

    def _create_pose_message(self, position: np.ndarray, quaternion: np.ndarray) -> Pose:
        """
        ROS Poseメッセージを作成
        
        Args:
            position: 位置ベクトル [x, y, z]
            quaternion: クォータニオン [x, y, z, w]
            
        Returns:
            ROS Poseメッセージ
        """
        pose_msg = Pose()
        
        # 位置設定
        pose_msg.position.x = float(position[0])
        pose_msg.position.y = float(position[1])
        pose_msg.position.z = float(position[2])
        
        # 姿勢設定（クォータニオン）
        pose_msg.orientation.x = float(quaternion[0])
        pose_msg.orientation.y = float(quaternion[1])
        pose_msg.orientation.z = float(quaternion[2])
        pose_msg.orientation.w = float(quaternion[3])
        
        return pose_msg

    def _smooth_pose(self, pose_msg: Pose) -> Pose:
        """
        姿勢の平滑化処理（移動平均）
        
        Args:
            pose_msg: 入力のPoseメッセージ
            
        Returns:
            平滑化されたPoseメッセージ
        """
        try:
            # 履歴に追加
            self.pose_history.append(pose_msg)
            
            # 履歴サイズを制限
            if len(self.pose_history) > self.history_size:
                self.pose_history.pop(0)
            
            # 平均値を計算
            if len(self.pose_history) == 1:
                return pose_msg
            
            # 位置の平均
            avg_x = sum(pose.position.x for pose in self.pose_history) / len(self.pose_history)
            avg_y = sum(pose.position.y for pose in self.pose_history) / len(self.pose_history)
            avg_z = sum(pose.position.z for pose in self.pose_history) / len(self.pose_history)
            
            # クォータニオンの平均（単純平均、より高度な方法もあり）
            avg_qx = sum(pose.orientation.x for pose in self.pose_history) / len(self.pose_history)
            avg_qy = sum(pose.orientation.y for pose in self.pose_history) / len(self.pose_history)
            avg_qz = sum(pose.orientation.z for pose in self.pose_history) / len(self.pose_history)
            avg_qw = sum(pose.orientation.w for pose in self.pose_history) / len(self.pose_history)
            
            # 正規化
            quat_norm = np.sqrt(avg_qx**2 + avg_qy**2 + avg_qz**2 + avg_qw**2)
            if quat_norm > 0:
                avg_qx /= quat_norm
                avg_qy /= quat_norm
                avg_qz /= quat_norm
                avg_qw /= quat_norm
            
            # 平滑化されたメッセージを作成
            smoothed_pose = Pose()
            smoothed_pose.position.x = avg_x
            smoothed_pose.position.y = avg_y
            smoothed_pose.position.z = avg_z
            smoothed_pose.orientation.x = avg_qx
            smoothed_pose.orientation.y = avg_qy
            smoothed_pose.orientation.z = avg_qz
            smoothed_pose.orientation.w = avg_qw
            
            return smoothed_pose
            
        except Exception:
            # 平滑化に失敗した場合は元のメッセージを返す
            return pose_msg

    def get_pose_info(self, pose_msg: Pose) -> str:
        """
        デバッグ用の姿勢情報文字列を生成
        
        Args:
            pose_msg: Poseメッセージ
            
        Returns:
            姿勢情報の文字列
        """
        return f"Position: ({pose_msg.position.x:.3f}, {pose_msg.position.y:.3f}, {pose_msg.position.z:.3f}), " \
               f"Orientation: ({pose_msg.orientation.x:.3f}, {pose_msg.orientation.y:.3f}, " \
               f"{pose_msg.orientation.z:.3f}, {pose_msg.orientation.w:.3f})"