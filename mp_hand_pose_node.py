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
try:
    from hand_control_interfaces.msg import MoveHand
    MOVE_HAND_MSG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MoveHand message: {e}")
    print("Hand control topic 無効化")
    MOVE_HAND_MSG_AVAILABLE = False
    # メッセージが定義されていない場合の代替実装
    class MoveHand:
        def __init__(self):
            self.id = 0  # uint8
            self.state = 79  # uint8: 79='O', 67='C'

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
        self.declare_parameter('model_type', 2)  # 0:lite, 1:full, 2:heavy
        self.declare_parameter('use_2D_visualization', True)
        self.declare_parameter('use_3D_visualization', False)
        self.declare_parameter('publish_rate', 60.0)  # Hz
        self.declare_parameter('hand_open_threshold', 0.4)  # 手の開閉判定閾値
        self.declare_parameter('dynamixel_id', 1)  # Dynamixel ID
        self.declare_parameter('enable_hand_control', True)  # hand_controlトピックの有効化
        self.declare_parameter('use_hand_model', True)  # MediaPipe Handsモデルの使用
        self.declare_parameter('fixed_orientation_planning', True)
        self.declare_parameter('use_plane_planning', True)  # 平面プランニングの使用
        self.declare_parameter('plane_planning_x', 0.3)  # 平面プランニングのX座標
        self.declare_parameter('coordinate_y_flip', True)  # Y座標反転

        # ========== 追加: ヒステリシス + 時間的連続性のパラメータ ==========
        self.declare_parameter('threshold_close_to_open', 0.1)  # 閉→開の閾値
        self.declare_parameter('threshold_open_to_close', 0.45)  # 開→閉の閾値
        self.declare_parameter('min_state_duration', 0.15)  # 状態変化の最小持続時間（秒）

        # パラメータの取得
        self.camera_device = self.get_parameter('camera_device').get_parameter_value().integer_value
        self.model_type = self.get_parameter('model_type').get_parameter_value().integer_value
        self.use_2D_visualization = self.get_parameter('use_2D_visualization').get_parameter_value().bool_value
        self.use_3D_visualization = self.get_parameter('use_3D_visualization').get_parameter_value().bool_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.hand_open_threshold = self.get_parameter('hand_open_threshold').get_parameter_value().double_value
        self.dynamixel_id = self.get_parameter('dynamixel_id').get_parameter_value().integer_value
        self.enable_hand_control = self.get_parameter('enable_hand_control').get_parameter_value().bool_value
        self.use_hand_model = self.get_parameter('use_hand_model').get_parameter_value().bool_value
        self.fixed_orientation_planning = self.get_parameter('fixed_orientation_planning').get_parameter_value().bool_value
        self.use_plane_planning = self.get_parameter('use_plane_planning').get_parameter_value().bool_value
        self.plane_planning_x = self.get_parameter('plane_planning_x').get_parameter_value().double_value
        self.threshold_close_to_open = self.get_parameter('threshold_close_to_open').get_parameter_value().double_value
        self.threshold_open_to_close = self.get_parameter('threshold_open_to_close').get_parameter_value().double_value
        self.min_state_duration = self.get_parameter('min_state_duration').get_parameter_value().double_value

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
        
        # hand_controlトピックのパブリッシャー
        self.hand_control_publisher = None
        if self.enable_hand_control and MOVE_HAND_MSG_AVAILABLE:
            try:
                self.hand_control_publisher = self.create_publisher(
                    MoveHand,
                    'hand_control',
                    10
                )
                self.get_logger().info('hand_control topic enabled with hand_control_interfaces/MoveHand message')
            except Exception as e:
                self.get_logger().error(f'Failed to create hand_control publisher: {e}')
                self.get_logger().info('Make sure hand_control_interfaces package is built and sourced')
                self.hand_control_publisher = None
                self.enable_hand_control = False
        elif self.enable_hand_control and not MOVE_HAND_MSG_AVAILABLE:
            self.get_logger().warn('hand_control enabled but MoveHand message not available')
            self.get_logger().warn('Please build hand_control_interfaces package first')
            self.enable_hand_control = False

        # 各機能クラスの初期化
        try:
            self.pose_calculator = PoseCalculator(
                hand_open_threshold=self.hand_open_threshold,
                fixed_orientation_planning=self.fixed_orientation_planning
            )
            
            self.video_processor = VideoProcessor(
                camera_device=self.camera_device,
                camera_width=self.camera_width,
                camera_height=self.camera_height,
                model_type=self.model_type,
                use_hand_model=self.use_hand_model,
                threshold_close_to_open=self.threshold_close_to_open,
                threshold_open_to_close=self.threshold_open_to_close,
                min_state_duration=self.min_state_duration
            )
            
            if self.use_2D_visualization or self.use_3D_visualization:
                self.visualizer = Visualizer(
                    enable_3d=self.use_3D_visualization
                )
            else:
                self.visualizer = None
                
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
        self.get_logger().info(f'  2D表示: {self.use_2D_visualization}')
        self.get_logger().info(f'  3D表示: {self.use_3D_visualization}')
        self.get_logger().info(f'  手の開閉判定閾値: {self.hand_open_threshold}')
        self.get_logger().info(f'  閉→開閾値: {self.threshold_close_to_open}')  # 追加
        self.get_logger().info(f'  開→閉閾値: {self.threshold_open_to_close}')  # 追加
        self.get_logger().info(f'  最小持続時間: {self.min_state_duration}秒')  # 追加
        self.get_logger().info(f'  Dynamixel ID: {self.dynamixel_id}')
        self.get_logger().info(f'  hand_control有効: {self.enable_hand_control}')
        self.get_logger().info(f'  Handsモデル使用: {self.use_hand_model}')
        self.get_logger().info(f'  Orientation固定: {self.fixed_orientation_planning}')
        self.get_logger().info(f'  平面プランニング使用: {self.use_plane_planning}')
        self.get_logger().info(f'  平面プランニングX: {self.plane_planning_x}')
        self.get_logger().info('='*50)
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
                # else:
                #     # Poseモデルから簡易的な開閉判定
                #     hand_status, confidence = self.pose_calculator.calculate_hand_status(
                #         pose_result.pose_world_landmarks[0]
                #     )
                
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