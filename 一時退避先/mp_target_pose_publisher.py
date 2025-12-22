#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy # type: ignore
from rclpy.node import Node # type: ignore
from std_msgs.msg import String # type: ignore
from geometry_msgs.msg import Pose # type: ignore

import os
import time
import copy
import argparse
from typing import List, Optional, Dict, Tuple, Union

import cv2
import numpy as np
import mediapipe as mp   # type: ignore
from mediapipe.tasks import python  # type: ignore
from mediapipe.tasks.python import vision   # type: ignore
import matplotlib.pyplot as plt # type: ignore
from utils import CvFpsCalc # type: ignore
from utils.download_file import download_file # type: ignore

# hand_control_interfacesパッケージからMoveHandメッセージをインポート
from hand_control_interfaces.msg import MoveHand # type: ignore

from .target_pose_calculator import TargetPoseCalculator
from .一時退避先.image_processor import ImageProcessor


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

        # モデル保存ディレクトリ
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))
        
        # モデルディレクトリを作成
        os.makedirs(model_dir, exist_ok=True)

        pose_model_path = self.model_downloader(pose_model_url, model_dir)
        hand_model_path = self.model_downloader(hand_model_url, model_dir)
        
        # 各機能の初期化
        try:
            # self.pose_calculator = TargetPoseCalculator(
            #     hand_open_threshold=self.hand_open_threshold,
            #     correct_inference_results=self.enable_hand_control,
            #     fixed_orientation_planning=self.fixed_orientation_planning
            # )    
            # FPS計測モジュール
            # cvFpsCalc: CvFpsCalc = CvFpsCalc(buffer_len=10) 
            # # カメラ焦点推定モジュール
            # self.camera_focal_point_estimator = CameraFocalPointEstimator()
            
            # # カメラ準備
            # cap: cv2.VideoCapture = cv2.VideoCapture([0,None])
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            # if not cap.isOpened():
            #     print("Error: Could not open camera")
            #     return
            
            # # PoseLandmarker生成
            # pose_base_options: python.BaseOptions = python.BaseOptions( model_asset_path=pose_model_path,) # type: ignore
            # pose_options: vision.PoseLandmarkerOptions = vision.PoseLandmarkerOptions(base_options=pose_base_options)# type: ignore
            # pose_detector: vision.PoseLandmarker = vision.PoseLandmarker.create_from_options(pose_options) # type: ignore
            # # HandLandmarker生成
            # hand_base_options: python.BaseOptions = python.BaseOptions(model_asset_path=hand_model_path) # type: ignore
            # hand_options: vision.HandLandmarkerOptions = vision.HandLandmarkerOptions( # type: ignore
            # base_options=hand_base_options,
            # num_hands=2,
            # )
            # hand_detector: vision.HandLandmarker = vision.HandLandmarker.create_from_options(hand_options)   # type: ignore
            
            # # World座標プロット準備
            # if self.use_3D_plot:
            #     fig = plt.figure()
            #     ax = fig.add_subplot(211, projection="3d")
            #     r_ax = fig.add_subplot(212, projection="3d")
            #     fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)
                
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

    def model_downloader(model_url: str, model_dir: str) -> str:
        """
        モデルファイルのパスを取得
        
        Args:
            model_url: モデルファイルのURL
            model_dir: モデル保存ディレクトリ
            
        Returns:
            モデルファイルの完全パス
        """

        # ダウンロードファイル名生成
        model_name: str = model_url.split('/')[-1]
        quantize_type: str = model_url.split('/')[-3]
        split_name: List[str] = model_name.split('.')
        model_filename: str = f"{split_name[0]}_{quantize_type}.{split_name[1]}"

        # 重みファイルダウンロード
        model_path: str = os.path.join(model_dir, model_filename)
        if not os.path.exists(model_path):
            print(f"Poseモデルファイルをダウンロード中: {model_url}")
            download_file(url=model_url, save_path=model_path)
            print(f"ダウンロード完了: {model_path}")

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
        # try:
            # フレーム取得と処理（ミラー処理込み）
            # 姿勢計算と手の状態判定
            # pose_msg = None
            # hand_status = "Unknown"
            # confidence = 0.0
            
            # Poseモデルから姿勢を計算
            
            # 平面プランニングを使用する場合
            # x座標を平面プランニングの値に設定
            # セーフティモードでない場合のみトピックを公開
                        # 手の開閉状態を公開
                        
                            # 状態が変化した場合のみ公開
                       
                    # 状態を保存（セーフティモードに関係なく更新）
                   

            # 可視化
           
                    # 2D描画（手の開閉状態を含む）
                   
                    # セーフティモードの状態を描画
                   
                    # 画面表示                
                # 3D描画
              
                # キー入力処理
             
                # 'ESCキーが押されました。ノードを終了します。
                       
                  
                #             self.get_logger().warn('セーフティモード: 有効 - トピック送信を停止しました')
                #             self.get_logger().info("'S'キーでセーフティモードを解除できます")
                    
         
                #             self.get_logger().info('セーフティモード: 解除 - トピック送信を再開しました')
                #             self.get_logger().info("'Space'キーでセーフティモードを有効化できます")


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