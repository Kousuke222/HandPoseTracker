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

        # 向き計算の履歴管理（タイムアウト処理用）
        self.last_valid_orientation = None
        self.last_orientation_time = None
        self.previous_quaternion = None

        # time モジュールのインポート
        import time
        self.time = time

    def calculate_and_convert_pose(self, pose_world_landmarks, hand_world_landmarks=None) -> Optional[Pose]:
        """
        MediaPipeのワールドランドマークからROS Poseメッセージを生成
        ミラー表示で固定

        Args:
            pose_world_landmarks: MediaPipeのワールドランドマーク（Poseモデル）
            hand_world_landmarks: MediaPipeのワールドランドマーク（Handsモデル、オプション）

        Returns:
            ROS Poseメッセージ（計算失敗時はNone）
        """
        try:
            # ランドマークの信頼度チェック
            if not self._validate_landmarks(pose_world_landmarks):
                return None

            # ランドマーク辞書を作成（ミラー表示固定）
            landmark_dict = self._create_landmark_dict(pose_world_landmarks)

            # 位置: Poseモデルの手首（インデックス15）を使用
            if 15 not in landmark_dict:
                return None
            wrist_position = np.array(landmark_dict[15])

            # 向き: Handsモデルがあれば最小二乗平面法、なければPoseモデルから計算
            quaternion = None
            if hand_world_landmarks is not None and not self.fixed_orientation_planning and self.config.use_hands_orientation:
                quaternion = self._calculate_hand_orientation_from_hands(hand_world_landmarks)

            # Handsモデルが使えない、または固定姿勢の場合
            if quaternion is None:
                if self.fixed_orientation_planning:
                    # 固定姿勢
                    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
                else:
                    # Poseモデルから計算（既存の方法）
                    _, quaternion, _ = self._calculate_hand_pose(landmark_dict)
                    if quaternion is None:
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

    def _calculate_hand_orientation_from_hands(self, hand_world_landmarks) -> Optional[np.ndarray]:
        """
        Handsモデルのランドマークから最小二乗平面法で向きを計算

        Args:
            hand_world_landmarks: Handsモデルのワールドランドマーク

        Returns:
            クォータニオン [x, y, z, w] または None（計算失敗時）
        """
        try:
            current_time = self.time.time()

            # Handsモデルが検出されていない場合のフォールバック
            if hand_world_landmarks is None:
                if self.last_valid_orientation is not None and self.last_orientation_time is not None:
                    time_since_last = current_time - self.last_orientation_time
                    if time_since_last < self.config.orientation_timeout:
                        # タイムアウト前: 前回値を保持
                        return self.last_valid_orientation
                # タイムアウト後: デフォルト姿勢
                return np.array([1.0, 0.0, 0.0, 0.0])

            # MCP関節のランドマークを収集（ミラー表示のためX座標反転）
            mcp_points = []
            mcp_indices_list = list(self.config.hand_mcp_indices.values())

            for idx in mcp_indices_list:
                if idx < len(hand_world_landmarks):
                    landmark = hand_world_landmarks[idx]
                    # ミラー表示のためX座標反転
                    point = np.array([
                        -landmark.x,  # ミラー反転
                        landmark.y,
                        landmark.z
                    ])
                    mcp_points.append(point)

            # 最低必要数のランドマークがあるかチェック
            if len(mcp_points) < self.config.min_mcp_landmarks:
                # フォールバック: 前回値または簡易法
                if self.last_valid_orientation is not None:
                    return self.last_valid_orientation
                return None

            # 最小二乗平面法で法線ベクトルを計算
            z_axis_candidate, planarity = self._least_squares_plane_normal(mcp_points)

            # 平面性が低い場合（点がほぼ一直線上）
            if planarity < self.config.planarity_threshold:
                if self.last_valid_orientation is not None:
                    return self.last_valid_orientation
                return None

            # 手首と中指先端を取得（向き判定用）
            if self.config.hand_wrist_index >= len(hand_world_landmarks) or \
               self.config.hand_middle_tip_index >= len(hand_world_landmarks):
                return None

            wrist = hand_world_landmarks[self.config.hand_wrist_index]
            middle_tip = hand_world_landmarks[self.config.hand_middle_tip_index]

            # ミラー表示対応
            wrist_pos = np.array([-wrist.x, wrist.y, wrist.z])
            middle_tip_pos = np.array([-middle_tip.x, middle_tip.y, middle_tip.z])

            # 法線の向きを判定（手のひら側を+Z方向に）
            finger_direction = middle_tip_pos - wrist_pos
            if np.dot(z_axis_candidate, finger_direction) < 0:
                z_axis = -z_axis_candidate
            else:
                z_axis = z_axis_candidate

            # X軸を計算（指先方向）
            x_axis = finger_direction / np.linalg.norm(finger_direction)

            # Y軸を計算（直交性を保証）
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)

            # X軸を再計算（完全な直交座標系にするため）
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)

            # MediaPipe座標系 -> ロボット座標系変換
            # MediaPipe: Y軸下向き, Z軸カメラ向き
            # ロボット: X軸前向き, Y軸左向き, Z軸上向き
            # 変換: [X, Y, Z]_mp -> [Z, X, -Y]_robot
            robot_x_axis = np.array([x_axis[2], x_axis[0], -x_axis[1]])
            robot_y_axis = np.array([y_axis[2], y_axis[0], -y_axis[1]])
            robot_z_axis = np.array([z_axis[2], z_axis[0], -z_axis[1]])

            # 回転行列を構築
            rotation_matrix = np.stack([robot_x_axis, robot_y_axis, robot_z_axis], axis=1)

            # クォータニオンに変換
            rot = R.from_matrix(rotation_matrix)
            quaternion = rot.as_quat()  # [x, y, z, w]形式

            # クォータニオンの連続性を保証
            if self.previous_quaternion is not None:
                quaternion = self._ensure_quaternion_continuity(quaternion, self.previous_quaternion)

            # SLERP補間による平滑化
            if self.config.use_slerp_smoothing and self.previous_quaternion is not None:
                quaternion = self._slerp(self.previous_quaternion, quaternion, self.config.slerp_alpha)

            # 状態更新
            self.last_valid_orientation = quaternion
            self.last_orientation_time = current_time
            self.previous_quaternion = quaternion

            return quaternion

        except Exception as e:
            print(f"Handsモデルによる向き計算エラー: {e}")
            # フォールバック: 前回値
            if self.last_valid_orientation is not None:
                return self.last_valid_orientation
            return None

    def _least_squares_plane_normal(self, points: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        最小二乗法により点群から平面の法線ベクトルを計算

        Args:
            points: 3次元点のリスト

        Returns:
            (法線ベクトル, 平面性スコア)
            平面性スコア = S[2] / S[0]（特異値の比率）
        """
        try:
            # 重心を計算
            points_array = np.array(points)
            centroid = np.mean(points_array, axis=0)

            # 各点から重心へのベクトル
            centered_points = points_array - centroid

            # 共分散行列を計算
            covariance_matrix = np.dot(centered_points.T, centered_points) / len(points)

            # SVD分解
            U, S, Vt = np.linalg.svd(covariance_matrix)

            # 最小固有値に対応する固有ベクトルが法線ベクトル
            normal_vector = Vt[-1]

            # 平面性スコア（特異値の比率）
            planarity_score = S[2] / S[0] if S[0] > 1e-6 else 0.0

            return normal_vector, planarity_score

        except Exception as e:
            print(f"最小二乗平面法エラー: {e}")
            return np.array([0, 0, 1]), 0.0

    def _slerp(self, quat1: np.ndarray, quat2: np.ndarray, alpha: float) -> np.ndarray:
        """
        球面線形補間（SLERP）

        Args:
            quat1: 開始クォータニオン [x, y, z, w]
            quat2: 終了クォータニオン [x, y, z, w]
            alpha: 補間パラメータ（0-1）

        Returns:
            補間されたクォータニオン [x, y, z, w]
        """
        try:
            # scipy.spatial.transform.Rotationを使用
            rot1 = R.from_quat(quat1)
            rot2 = R.from_quat(quat2)

            # SLERPキーフレーム
            key_times = [0, 1]
            key_rots = R.from_quat([quat1, quat2])

            # Slerpを使用
            from scipy.spatial.transform import Slerp
            slerp = Slerp(key_times, key_rots)
            interpolated_rot = slerp([alpha])[0]

            return interpolated_rot.as_quat()

        except Exception as e:
            print(f"SLERP補間エラー: {e}")
            return quat2

    def _ensure_quaternion_continuity(self, current_quat: np.ndarray, previous_quat: np.ndarray) -> np.ndarray:
        """
        クォータニオンの符号反転をチェックして連続性を保証

        Args:
            current_quat: 現在のクォータニオン
            previous_quat: 前回のクォータニオン

        Returns:
            符号調整されたクォータニオン
        """
        # クォータニオンのジャンプ防止
        if np.dot(current_quat, previous_quat) < 0:
            return -current_quat
        return current_quat