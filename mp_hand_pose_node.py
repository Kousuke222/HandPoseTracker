#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import cv2
import copy
from typing import Optional, Tuple

# hand_control_interfacesパッケージからMoveHandメッセージをインポート
from hand_control_interfaces.msg import MoveHand

# 自作モジュールのインポート
from .pose_calculator import PoseCalculator
from .video_processor import VideoProcessor
from .visualizer import Visualizer
from .config import HandPoseConfig


class HandPosePublisher(Node):
    """
    MediaPipeを使用した右手姿勢検出・公開を行うROS2ノード
    PoseモデルとHandsモデルの両方を使用
    """

    def __init__(self):
        super().__init__('mp_hand_pose_publisher')
        
        # パラメータを個別に宣言（ROS2 Humbleでの推奨方法）
        self.declare_parameter('camera_device', 0)
        self.declare_parameter('publish_rate', 60.0)  # Hz
        self.declare_parameter('hand_open_threshold', 0.4)  # 手の開閉判定閾値
        self.declare_parameter('dynamixel_id', 1)  # Dynamixel ID
        self.declare_parameter('fixed_orientation_planning', True)
        self.declare_parameter('use_plane_planning', True)  # 平面プランニングの使用
        self.declare_parameter('plane_planning_x', 0.3)  # 平面プランニングのX座標
        self.declare_parameter('coordinate_y_flip', True)  # Y座標反転
        self.declare_parameter('threshold_close_to_open', 0.1)  # 閉→開の閾値
        self.declare_parameter('threshold_open_to_close', 0.45)  # 開→閉の閾値
        self.declare_parameter('min_state_duration', 0.15)  # 状態変化の最小持続時間（秒）
        self.declare_parameter('use_hands_orientation', True)  # Handsモデルによる向き計算の有効/無効

        # パラメータの取得
        self.camera_device = self.get_parameter('camera_device').get_parameter_value().integer_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.hand_open_threshold = self.get_parameter('hand_open_threshold').get_parameter_value().double_value
        self.dynamixel_id = self.get_parameter('dynamixel_id').get_parameter_value().integer_value
        self.fixed_orientation_planning = self.get_parameter('fixed_orientation_planning').get_parameter_value().bool_value
        self.use_plane_planning = self.get_parameter('use_plane_planning').get_parameter_value().bool_value
        self.plane_planning_x = self.get_parameter('plane_planning_x').get_parameter_value().double_value
        self.coordinate_y_flip = self.get_parameter('coordinate_y_flip').get_parameter_value().bool_value
        self.threshold_close_to_open = self.get_parameter('threshold_close_to_open').get_parameter_value().double_value
        self.threshold_open_to_close = self.get_parameter('threshold_open_to_close').get_parameter_value().double_value
        self.min_state_duration = self.get_parameter('min_state_duration').get_parameter_value().double_value
        self.use_hands_orientation = self.get_parameter('use_hands_orientation').get_parameter_value().bool_value

        # 固定値設定
        self.camera_width = 640
        self.camera_height = 480

        # 安全機能関連の初期化
        self.safety_mode = True  # セーフティモードフラグ（True: トピック送信停止）
        self.safety_mode_changed = False  # モード変更フラグ（画面更新用）

        # パブリッシャーの作成
        self.pose_publisher = self.create_publisher(
            Pose, 
            'target_pose', 
            10
        )
        
        # hand_controlトピックのパブリッシャー（常に有効）
        self.hand_control_publisher = self.create_publisher(
            MoveHand,
            'hand_control',
            10
        )
        self.get_logger().info('hand_control topic enabled')

        # 各機能クラスの初期化
        try:
            self.pose_calculator = PoseCalculator(
                hand_open_threshold=self.hand_open_threshold,
                fixed_orientation_planning=self.fixed_orientation_planning
            )
            # ROSパラメータをconfigに反映
            self.pose_calculator.config.use_hands_orientation = self.use_hands_orientation

            self.video_processor = VideoProcessor(
                camera_device=self.camera_device,
                camera_width=self.camera_width,
                camera_height=self.camera_height,
                threshold_close_to_open=self.threshold_close_to_open,
                threshold_open_to_close=self.threshold_open_to_close,
                min_state_duration=self.min_state_duration
            )

            # 2D可視化は常に有効
            self.visualizer = Visualizer()
                
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
        self.get_logger().info(f'  カメラデバイス: {self.camera_device}')
        self.get_logger().info(f'  解像度: {self.camera_width}x{self.camera_height}')
        self.get_logger().info(f'  Poseモデル: heavy（固定）')
        self.get_logger().info(f'  Handsモデル: 有効（固定）')
        self.get_logger().info(f'  2D表示: 有効（固定）')
        self.get_logger().info(f'  hand_control: 有効（固定）')
        self.get_logger().info(f'  手の開閉判定閾値: {self.hand_open_threshold}')
        self.get_logger().info(f'  閉→開閾値: {self.threshold_close_to_open}')
        self.get_logger().info(f'  開→閉閾値: {self.threshold_open_to_close}')
        self.get_logger().info(f'  最小持続時間: {self.min_state_duration}秒')
        self.get_logger().info(f'  Dynamixel ID: {self.dynamixel_id}')
        self.get_logger().info(f'  Orientation固定: {self.fixed_orientation_planning}')
        self.get_logger().info(f'  Handsモデルによる手の向き計算: {self.use_hands_orientation}')
        self.get_logger().info(f'  平面プランニング使用: {self.use_plane_planning}')
        self.get_logger().info(f'  平面プランニングX: {self.plane_planning_x}')
        self.get_logger().info(f'  Y座標反転: {self.coordinate_y_flip}')
        self.get_logger().info('='*50)
        self.get_logger().info('キー操作:')
        self.get_logger().info('  Space: セーフティモード（トピック送信停止）')
        self.get_logger().info('  S: セーフティモード解除')
        self.get_logger().info('  ESC: プログラム終了')
        self.get_logger().info('='*50)

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
                # Handsモデルの結果を取得（利用可能な場合）
                hand_world_landmarks = None
                if hand_result and hand_result['right_hand']['world_landmarks']:
                    hand_world_landmarks = hand_result['right_hand']['world_landmarks']

                # 位置（Pose）+ 向き（Hands）で姿勢を計算
                pose_msg = self.pose_calculator.calculate_and_convert_pose(
                    pose_result.pose_world_landmarks[0],
                    hand_world_landmarks=hand_world_landmarks
                )

                # Handsモデルから開閉判定
                if hand_result and hand_result['right_hand']['world_landmarks']:
                    hand_status, confidence = self.video_processor.hand_detector.calculate_hand_openness(
                        hand_result['right_hand']['world_landmarks'],
                        threshold=self.hand_open_threshold
                    )
                
                if pose_msg:
                    if self.use_plane_planning:
                        # 平面プランニングを使用する場合
                        # x座標を平面プランニングの値に設定
                        pose_msg.position.x = self.plane_planning_x

                    if self.coordinate_y_flip:
                        # Y座標を反転
                        pose_msg.position.y = -pose_msg.position.y
                        
                    self.last_valid_pose = pose_msg
                    
                    # セーフティモードでない場合のみトピックを公開
                    if not self.safety_mode:
                        # 姿勢の公開
                        self.pose_publisher.publish(pose_msg)

                        # 手の開閉状態を公開（常に有効）
                        if self.hand_control_publisher and hand_status in ['O', 'C']:
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
                            f"[Using Hands model]"
                        )

            # 可視化（常に有効）
            if self.visualizer:
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

                # キー入力処理
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
                    
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.get_logger().error(f'フレーム処理エラー: {e}', throttle_duration_sec=1.0)

    def destroy_node(self):
        """
        ノード終了時のクリーンアップ
        """
        self.get_logger().info('ノードを終了しています...')
        try:
            if hasattr(self, 'video_processor') and self.video_processor:
                self.video_processor.cleanup()
            if hasattr(self, 'visualizer') and self.visualizer:
                self.visualizer.cleanup()
            cv2.destroyAllWindows()
        except Exception as e:
            self.get_logger().error(f'クリーンアップエラー: {e}')
        
        super().destroy_node()


def main(args=None):
    """
    メイン関数
    """
    print('MediaPipe Hand Pose Publisher を開始します...')
    
    try:
        rclpy.init(args=args)
        hand_pose_publisher = HandPosePublisher()
        rclpy.spin(hand_pose_publisher)
        
    except KeyboardInterrupt:
        print('\nCtrl+Cが押されました。終了します。')
    except Exception as e:
        print(f'エラーが発生しました: {e}')
        import traceback
        traceback.print_exc()
    finally:
        try:
            if 'hand_pose_publisher' in locals():
                hand_pose_publisher.destroy_node()
        except:
            pass
        
        if rclpy.ok():
            rclpy.shutdown()
        
        print('MediaPipe Hand Pose Publisher を終了しました。')


if __name__ == '__main__':
    main()