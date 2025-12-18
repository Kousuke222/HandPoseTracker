#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose

import os
import time
import copy
import argparse
from typing import List, Optional, Dict, Tuple, Union

import cv2
import numpy as np
import mediapipe as mp  
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision  
import matplotlib.pyplot as plt
from utils import CvFpsCalc
from utils.download_file import download_file

# hand_control_interfacesパッケージからMoveHandメッセージをインポート
from hand_control_interfaces.msg import MoveHand

from .target_pose_calculator import TargetPoseCalculator
from .image_processor import ImageProcessor


class TargetPosePublisher(Node):
    """
    MediaPipeを使用した右手姿勢検出・公開を行うROS2ノード
    PoseモデルとHandsモデルの両方を使用
    """

    def __init__(self):
        super().__init__('mp_hand_pose_publisher')
        
        self.declare_parameter('camera_device', 0)
        self.declare_parameter('camera_width', 960)
        self.declare_parameter('camera_height', 540)
        self.declare_parameter('use_2D_plot', True)
        self.declare_parameter('use_3D_plot', True)
        self.declare_parameter('publish_rate', 60.0)  # Hz
        self.declare_parameter('hand_open_threshold', 0.4)  # 手の開閉判定閾値
        self.declare_parameter('dynamixel_id', 1)  # Dynamixel ID
        self.declare_parameter('correct_inference_results', True)  # 推論結果補正アルゴリズム
        self.declare_parameter('fixed_orientation_planning', True) # 固定姿勢プランニング
        self.declare_parameter('use_plane_planning', True)  # 平面プランニングの使用
        self.declare_parameter('plane_planning_x', 0.3)  # 平面プランニングのX座標

        # パラメータの取得
        self.camera_width = self.get_parameter('camera_width').get_parameter_value().integer_value
        self.camera_height = self.get_parameter('camera_height').get_parameter_value().integer_value
        self.use_2D_visualization = self.get_parameter('use_2D_plot').get_parameter_value().bool_value
        self.use_3D_visualization = self.get_parameter('use_3D_plot').get_parameter_value().bool_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.hand_open_threshold = self.get_parameter('hand_open_threshold').get_parameter_value().double_value
        self.dynamixel_id = self.get_parameter('dynamixel_id').get_parameter_value().integer_value
        self.enable_hand_control = self.get_parameter('correct_inference_results').get_parameter_value().bool_value
        self.fixed_orientation_planning = self.get_parameter('fixed_orientation_planning').get_parameter_value().bool_value
        self.use_plane_planning = self.get_parameter('use_plane_planning').get_parameter_value().bool_value
        self.plane_planning_x = self.get_parameter('plane_planning_x').get_parameter_value().double_value

        # 安全機能関連の初期化
        self.safety_mode = True  # セーフティモードフラグ（True: トピック送信停止）
        self.safety_mode_changed = False  # モード変更フラグ（画面更新用）

        # パブリッシャーの作成
        self.pose_publisher = self.create_publisher(
            Pose, 
            'target_pose', 
            10
        )
        
        # hand_controlトピックのパブリッシャー
        self.hand_control_publisher = self.create_publisher(
            MoveHand,
            'hand_control',
            10
        )

        pose_model_url: str= 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task'
        hand_model_url: str= 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'

        pose_model_path = self.model_downloader(pose_model_url)
        hand_model_path = self.model_downloader(hand_model_url)
        
        # 各機能の初期化
        try:
            # self.pose_calculator = TargetPoseCalculator(
            #     hand_open_threshold=self.hand_open_threshold,
            #     correct_inference_results=self.enable_hand_control,
            #     fixed_orientation_planning=self.fixed_orientation_planning
            # )    
            # FPS計測モジュール
            cvFpsCalc: CvFpsCalc = CvFpsCalc(buffer_len=10) 
            # カメラ焦点推定モジュール
            self.camera_focal_point_estimator = CameraFocalPointEstimator()
            
            # カメラ準備
            cap: cv2.VideoCapture = cv2.VideoCapture([0,None])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            if not cap.isOpened():
                print("Error: Could not open camera")
                return
            
            # PoseLandmarker生成
            pose_base_options: python.BaseOptions = python.BaseOptions( model_asset_path=pose_model_path,) # type: ignore
            pose_options: vision.PoseLandmarkerOptions = vision.PoseLandmarkerOptions(base_options=pose_base_options)# type: ignore
            pose_detector: vision.PoseLandmarker = vision.PoseLandmarker.create_from_options(pose_options) # type: ignore
            # HandLandmarker生成
            hand_base_options: python.BaseOptions = python.BaseOptions(model_asset_path=hand_model_path) # type: ignore
            hand_options: vision.HandLandmarkerOptions = vision.HandLandmarkerOptions( # type: ignore
            base_options=hand_base_options,
            num_hands=2,
            )
            hand_detector: vision.HandLandmarker = vision.HandLandmarker.create_from_options(hand_options)   # type: ignore
            
            # World座標プロット準備
            if self.use_3D_plot:
                fig = plt.figure()
                ax = fig.add_subplot(211, projection="3d")
                r_ax = fig.add_subplot(212, projection="3d")
                fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)
                
        except Exception as e:
            self.get_logger().error(f'初期化エラー: {e}')
            raise

        # タイマーの作成（指定レートでコールバック実行）
        timer_period = 1.0 / self.publish_rate  # 秒
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # 状態変数
        self.frame_count = 0
        self.last_valid_pose: Optional[Pose] = None
        self.last_hand_state: Optional[str] = None
        self.hand_state_confidence: float = 0.0
        
        self.get_logger().info('='*50)
        self.get_logger().info('Hand Pose Publisher ノードが開始されました')
        self.get_logger().info('='*50)
        self.get_logger().info(f'設定:')
        self.get_logger().info(f'  公開レート: {self.publish_rate} Hz')
        self.get_logger().info(f'  解像度: {self.camera_width}x{self.camera_height}')
        self.get_logger().info(f'  2D表示: {self.use_2D_plot}')
        self.get_logger().info(f'  3D表示: {self.use_3D_plot}')
        self.get_logger().info(f'  手の開閉判定閾値: {self.hand_open_threshold}')
        self.get_logger().info(f'  Dynamixel ID: {self.dynamixel_id}')
        self.get_logger().info(f'  推論結果補正: {self.correct_inference_results}')
        self.get_logger().info(f'  Orientation固定: {self.fixed_orientation_planning}')
        self.get_logger().info(f'  平面プランニング使用: {self.use_plane_planning}')
        self.get_logger().info(f'  平面プランニングX: {self.plane_planning_x}')
        self.get_logger().info('='*50)
        self.get_logger().info('キー操作:')
        self.get_logger().info('  Space: セーフティモード（トピック送信停止）')
        self.get_logger().info('  S: セーフティモード解除')
        self.get_logger().info('  ESC: プログラム終了')
        self.get_logger().info('='*50)

    def model_downloader(model_url: str) -> str:
        # ダウンロードファイル名生成
        model_name: str = model_url.split('/')[-1]
        quantize_type: str = model_url.split('/')[-3]
        split_name: List[str] = model_name.split('.')
        model_name = split_name[0] + '_' + quantize_type + '.' + split_name[1]

        # 重みファイルダウンロード
        model_path: str = os.path.join('model', model_name)
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            download_file(url=model_url, save_path=model_path)

        return model_path
    
    def draw_safety_status(self, image):
        """
        セーフティモードの状態を画像に描画
        
        Args:
            image: 描画対象の画像
        """
        if self.safety_mode:
            # セーフティモード中は赤い警告表示
            cv2.rectangle(image, (10, 100), (300, 140), (0, 0, 255), -1)
            cv2.putText(
                image, "SAFETY MODE: ON", (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                image, "Press 'S' to resume", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2, cv2.LINE_AA
            )
        else:
            # 通常モードは緑色表示
            cv2.rectangle(image, (10, 100), (200, 130), (0, 255, 0), 2)
            cv2.putText(
                image, "ACTIVE", (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2, cv2.LINE_AA
            )



    

    def timer_callback(self):
        """
        メインのコールバック関数：フレーム処理と姿勢公開
        """
        try:
            # フレーム取得と処理（ミラー処理込み）
            frame, pose_result, hand_result = self.video_processor.process_frame()
            
            if frame is None:
                if self.frame_count % 100 == 0:  # 100フレームごとに警告
                    self.get_logger().warn('フレーム取得に失敗しました')
                return
            
            self.frame_count += 1
            
            # 姿勢計算と手の状態判定
            pose_msg = None
            hand_status = "Unknown"
            confidence = 0.0
            
            # Poseモデルから姿勢を計算
            if pose_result and len(pose_result.pose_world_landmarks) > 0:
                pose_msg = self.pose_calculator.calculate_and_convert_pose(
                    pose_result.pose_world_landmarks[0]
                )
                
                # Handsモデルが利用可能な場合はそちらを優先
                if self.use_hand_model and hand_result and hand_result['right_hand']['world_landmarks']:
                    # Handsモデルから高精度な開閉判定
                    hand_status, confidence = self.video_processor.hand_detector.calculate_hand_openness(
                        hand_result['right_hand']['world_landmarks'],
                        threshold=self.hand_open_threshold
                    )
                else:
                    # Poseモデルから簡易的な開閉判定
                    hand_status, confidence = self.pose_calculator.calculate_hand_status(
                        pose_result.pose_world_landmarks[0]
                    )
                
                if pose_msg:
                    if self.use_plane_planning:
                        # 平面プランニングを使用する場合
                        # x座標を平面プランニングの値に設定
                        pose_msg.position.x = self.plane_planning_x

                    self.last_valid_pose = pose_msg
                    
                    # セーフティモードでない場合のみトピックを公開
                    if not self.safety_mode:
                        # 姿勢の公開
                        self.pose_publisher.publish(pose_msg)
                        
                        # 手の開閉状態を公開
                        if self.enable_hand_control and self.hand_control_publisher and hand_status in ['O', 'C']:
                            # 状態が変化した場合のみ公開
                            if hand_status != self.last_hand_state:
                                hand_msg = MoveHand()
                                hand_msg.id = int(self.dynamixel_id)  # uint8に変換
                                hand_msg.state = ord(hand_status)  # 'O'->79, 'C'->67 (ASCII値)
                                self.hand_control_publisher.publish(hand_msg)
                                
                                self.get_logger().info(
                                    f'Hand control command sent: {hand_status} (ASCII: {ord(hand_status)}) '
                                    f'[ID: {self.dynamixel_id}, Confidence: {confidence:.3f}]'
                                )
                                self.last_hand_state = hand_status
                    
                    # 状態を保存（セーフティモードに関係なく更新）
                    self.hand_state_confidence = confidence
                        
                    if self.frame_count % 30 == 0:  # 1秒おきにログ出力（30FPSの場合）
                        if self.safety_mode:
                            self.get_logger().info(f'[SAFETY MODE] 手の姿勢を検出中（送信停止）: Frame {self.frame_count}')
                        else:
                            self.get_logger().info(f'手の姿勢を検出・公開中: Frame {self.frame_count}')
                        
                        self.get_logger().info(
                            f"Position: ({pose_msg.position.x:.3f}, {pose_msg.position.y:.3f}, {pose_msg.position.z:.3f})"
                        )
                        self.get_logger().info(
                            f"Orientation: ({pose_msg.orientation.x:.3f}, {pose_msg.orientation.y:.3f}, "
                            f"{pose_msg.orientation.z:.3f}, {pose_msg.orientation.w:.3f})"
                        )
                        self.get_logger().info(
                            f"Hand Status: {hand_status} (Confidence: {confidence:.3f}) "
                            f"[Using {'Hands' if self.use_hand_model and hand_result else 'Pose'} model]"
                        )

            # 可視化
            if self.visualizer:
                if self.use_2D_visualization:
                    # 2D描画（手の開閉状態を含む）
                    debug_image = copy.deepcopy(frame)
                    debug_image = self.visualizer.draw_2d_landmarks_with_hands(
                        debug_image, 
                        pose_result,
                        hand_result,
                        self.video_processor.get_fps(),
                        hand_status=hand_status,
                        hand_confidence=confidence
                    )
                    
                    # セーフティモードの状態を描画
                    self.draw_safety_status(debug_image)
                    
                    # 画面表示
                    cv2.imshow('MediaPipe Hand Pose Detection', debug_image)
                
                # 3D描画（Handsモデルの結果も含む）
                if self.use_3D_visualization and pose_result and len(pose_result.pose_world_landmarks) > 0:
                    pose_data = None
                    if pose_msg:
                        pose_data = {
                            'position': (pose_msg.position.x, pose_msg.position.y, pose_msg.position.z),
                            'quaternion': (pose_msg.orientation.x, pose_msg.orientation.y, 
                                         pose_msg.orientation.z, pose_msg.orientation.w)
                        }
                    
                    # Handsモデルのデータがある場合は3D表示に追加
                    hand_3d_data = None
                    if hand_result and hand_result['right_hand']['world_landmarks']:
                        hand_3d_data = hand_result['right_hand']['world_landmarks']
                    
                    self.visualizer.draw_3d_landmarks_with_hands(
                        pose_result,
                        pose_data,
                        hand_3d_data
                    )
                
                # キー入力処理
                if self.use_2D_visualization:
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == 27:  # ESC
                        self.get_logger().info('ESCキーが押されました。ノードを終了します。')
                        raise KeyboardInterrupt
                    
                    elif key == ord(' '):  # Space - セーフティモード有効化
                        if not self.safety_mode:
                            self.safety_mode = True
                            self.safety_mode_changed = True
                            self.get_logger().warn('セーフティモード: 有効 - トピック送信を停止しました')
                            self.get_logger().info("'S'キーでセーフティモードを解除できます")
                    
                    elif key == ord('s') or key == ord('S'):  # S - セーフティモード解除
                        if self.safety_mode:
                            self.safety_mode = False
                            self.safety_mode_changed = True
                            self.get_logger().info('セーフティモード: 解除 - トピック送信を再開しました')
                            self.get_logger().info("'Space'キーでセーフティモードを有効化できます")
                    
                    elif key == ord('1'):  # 1 - 固定位置へ移動
                        self.send_fixed_position(0.3, 0.0, 0.45)
                        if self.safety_mode:
                            self.get_logger().info('（セーフティモード中ですが、固定位置コマンドは送信されました）')
                    
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.get_logger().error(f'フレーム処理エラー: {e}', throttle_duration_sec=1.0)

    def destroy_node(self):
        """
        ノード終了時のクリーンアップ
        """
        self.get_logger().info('ノードを終了しています...')
        # try:
        #     if hasattr(self, 'video_processor') and self.video_processor:
        #         self.video_processor.cleanup()
        #     if hasattr(self, 'visualizer') and self.visualizer:
        #         self.visualizer.cleanup()
        #     cv2.destroyAllWindows()
        # except Exception as e:
        #     self.get_logger().error(f'クリーンアップエラー: {e}')
        
        super().destroy_node()



def main(args=None):
    """
    メイン関数
    """
    print('MediaPipe Target Pose Publisher を開始します...')
    
    try:
        rclpy.init(args=args)
        target_pose_publisher = TargetPosePublisher()
        rclpy.spin(target_pose_publisher)
        
    except KeyboardInterrupt:
        print('\nCtrl+Cが押されました。終了します。')
    except Exception as e:
        print(f'エラーが発生しました: {e}')
        import traceback
        traceback.print_exc()
    finally:
        try:
            if 'target_pose_publisher' in locals():
                target_pose_publisher.destroy_node()
        except:
            pass
        
        if rclpy.ok():
            rclpy.shutdown()
        
        print('MediaPipe Target Pose Publisher を終了しました。')


if __name__ == '__main__':
    main()