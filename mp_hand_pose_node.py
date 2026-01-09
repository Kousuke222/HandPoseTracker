#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import cv2
import copy
import time
from typing import Optional, Tuple

# hand_control_interfacesパッケージからMoveHandメッセージをインポート
from hand_control_interfaces.msg import MoveHand

# 自作モジュールのインポート
from .pose_calculator import PoseCalculator
from .video_processor import VideoProcessor
from .visualizer import Visualizer
from .config import HandPoseConfig
from .depth_estimator import DepthEstimator


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
        self.declare_parameter('use_plane_planning', False)  # 平面プランニングの使用
        self.declare_parameter('plane_planning_x', 0.3)  # 平面プランニングのX座標
        self.declare_parameter('coordinate_y_flip', True)  # Y座標反転
        self.declare_parameter('threshold_close_to_open', 0.1)  # 閉→開の閾値
        self.declare_parameter('threshold_open_to_close', 0.45)  # 開→閉の閾値
        self.declare_parameter('min_state_duration', 0.15)  # 状態変化の最小持続時間（秒）
        self.declare_parameter('use_depth_estimation', True)  # 深度推定の使用
        self.declare_parameter('depth_window_radius', 5)  # 深度取得の円状領域の半径（ピクセル）
        # 深度マッピングパラメータ
        self.declare_parameter('depth_map_min', 2.0)  # 深度マップの最小値
        self.declare_parameter('depth_map_max', 8.0)  # 深度マップの最大値
        self.declare_parameter('real_depth_min', 0.05)  # 実際の距離の最小値（m）
        self.declare_parameter('real_depth_max', 1.5)  # 実際の距離の最大値（m）
        self.declare_parameter('depth_offset_forward', -0.1)  # 深度補正値の前方へのオフセット (m)

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
        self.use_depth_estimation = self.get_parameter('use_depth_estimation').get_parameter_value().bool_value
        self.depth_window_radius = self.get_parameter('depth_window_radius').get_parameter_value().integer_value
        self.depth_map_min = self.get_parameter('depth_map_min').get_parameter_value().double_value
        self.depth_map_max = self.get_parameter('depth_map_max').get_parameter_value().double_value
        self.real_depth_min = self.get_parameter('real_depth_min').get_parameter_value().double_value
        self.real_depth_max = self.get_parameter('real_depth_max').get_parameter_value().double_value
        self.depth_offset_forward = self.get_parameter('depth_offset_forward').get_parameter_value().double_value

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

            # 深度推定器の初期化
            self.depth_estimator = None
            if self.use_depth_estimation:
                try:
                    self.depth_estimator = DepthEstimator(device='cuda')
                    self.get_logger().info('深度推定器を初期化しました')
                except Exception as depth_error:
                    self.get_logger().error(f'深度推定器の初期化に失敗しました: {depth_error}')
                    self.get_logger().warn('深度推定なしで続行します')
                    self.use_depth_estimation = False

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

        # 深度推定関連の状態変数
        self.last_wrist_depth: Optional[float] = None
        self.depth_estimation_time: float = 0.0
        self.mediapipe_time: float = 0.0
        self.last_original_x: Optional[float] = None
        self.last_corrected_x: Optional[float] = None

        # raw_depthの最小値・最大値の記録（キャリブレーション用）
        self.recorded_depth_min: Optional[float] = None
        self.recorded_depth_max: Optional[float] = None
        
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
        self.get_logger().info(f'  平面プランニング使用: {self.use_plane_planning}')
        self.get_logger().info(f'  平面プランニングX: {self.plane_planning_x}')
        self.get_logger().info(f'  Y座標反転: {self.coordinate_y_flip}')
        self.get_logger().info(f'  深度推定使用: {self.use_depth_estimation}')
        if self.use_depth_estimation:
            self.get_logger().info(f'  深度取得窓半径: {self.depth_window_radius}px (円状)')
            self.get_logger().info(f'  深度マップ範囲: {self.depth_map_min} ～ {self.depth_map_max}')
            self.get_logger().info(f'  実距離範囲: {self.real_depth_min}m ～ {self.real_depth_max}m')
        self.get_logger().info('='*50)
        self.get_logger().info('キー操作:')
        self.get_logger().info('  Space: セーフティモード（トピック送信停止）')
        self.get_logger().info('  S: セーフティモード解除')
        self.get_logger().info('  ESC: プログラム終了')
        self.get_logger().info('='*50)

    def get_wrist_pixel_coords(self, pose_result) -> Optional[Tuple[int, int]]:
        """
        MediaPipe Poseから右手首の画像座標を取得

        Args:
            pose_result: MediaPipeのPose検出結果

        Returns:
            (x, y): 画像座標（ピクセル単位）、取得失敗時はNone
        """
        try:
            if not pose_result or not pose_result.pose_landmarks:
                return None

            # 右手首のランドマークインデックスは15
            wrist_landmark = pose_result.pose_landmarks[0][15]

            # 正規化座標（0-1）をピクセル座標に変換
            wrist_x = int(wrist_landmark.x * self.camera_width)
            wrist_y = int(wrist_landmark.y * self.camera_height)

            # 画像範囲内かチェック
            if 0 <= wrist_x < self.camera_width and 0 <= wrist_y < self.camera_height:
                return (wrist_x, wrist_y)
            else:
                return None

        except Exception as e:
            self.get_logger().error(f'手首座標取得エラー: {e}', throttle_duration_sec=5.0)
            return None

    def convert_depth_to_real_distance(self, depth_value: float) -> Optional[float]:
        """
        深度マップの値を実際の距離（メートル）に変換

        Args:
            depth_value: 深度マップから取得した値

        Returns:
            実際の距離（メートル）、変換失敗時はNone
        """
        try:
            # 深度マップの値域チェック
            if depth_value < self.depth_map_min or depth_value > self.depth_map_max:
                # 範囲外の場合はクランプ
                depth_value = max(self.depth_map_min, min(self.depth_map_max, depth_value))

            # 正規化 (0-1)
            normalized = (depth_value - self.depth_map_min) / (self.depth_map_max - self.depth_map_min)

            # 実際の距離に変換
            real_distance = self.real_depth_min + normalized * (self.real_depth_max - self.real_depth_min)

            return float(real_distance)

        except Exception as e:
            self.get_logger().error(f'深度変換エラー: {e}', throttle_duration_sec=5.0)
            return None

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
            # MediaPipe処理時間の計測開始
            mediapipe_start = time.time()

            # フレーム取得と処理（ミラー処理込み）
            frame, pose_result, hand_result = self.video_processor.process_frame()

            if frame is None:
                if self.frame_count % 100 == 0:  # 100フレームごとに警告
                    self.get_logger().warn('フレーム取得に失敗しました')
                return

            # MediaPipe処理時間の計測終了
            self.mediapipe_time = time.time() - mediapipe_start

            self.frame_count += 1

            # 深度推定の実行（並列処理負荷評価のため）
            depth_map = None
            wrist_depth = None
            if self.use_depth_estimation and self.depth_estimator:
                depth_start = time.time()
                depth_map = self.depth_estimator.predict(frame)
                self.depth_estimation_time = time.time() - depth_start

                # 右手首の深度値を取得
                if depth_map is not None:
                    wrist_coords = self.get_wrist_pixel_coords(pose_result)
                    if wrist_coords is not None:
                        wrist_x, wrist_y = wrist_coords
                        wrist_depth = self.depth_estimator.get_circular_depth(
                            depth_map,
                            wrist_x,
                            wrist_y,
                            self.depth_window_radius
                        )
                        if wrist_depth is not None:
                            self.last_wrist_depth = wrist_depth

                            # 記録された最小値・最大値を更新
                            if self.recorded_depth_min is None or wrist_depth < self.recorded_depth_min:
                                self.recorded_depth_min = wrist_depth
                            if self.recorded_depth_max is None or wrist_depth > self.recorded_depth_max:
                                self.recorded_depth_max = wrist_depth

            # 姿勢計算と手の状態判定
            pose_msg = None
            hand_status = "Unknown"
            confidence = 0.0

            # Poseモデルから姿勢を計算
            if pose_result and len(pose_result.pose_world_landmarks) > 0:
                pose_msg = self.pose_calculator.calculate_and_convert_pose(
                    pose_result.pose_world_landmarks[0]
                )

                # Handsモデルから高精度な開閉判定（常に使用）
                if hand_result and hand_result['right_hand']['world_landmarks']:
                    hand_status, confidence = self.video_processor.hand_detector.calculate_hand_openness(
                        hand_result['right_hand']['world_landmarks'],
                        threshold=self.hand_open_threshold
                    )
                
                if pose_msg:
                    # 深度推定による奥行き補正（平面プランニング無効時のみ）
                    self.last_original_x = pose_msg.position.x
                    self.last_corrected_x = None
                    if self.use_depth_estimation and not self.use_plane_planning and self.last_wrist_depth is not None:
                        self.last_corrected_x = self.convert_depth_to_real_distance(self.last_wrist_depth)
                        if self.last_corrected_x is not None:
                            pose_msg.position.x = self.last_corrected_x + self.depth_offset_forward

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
                        
                    if self.frame_count % 5 == 0:  # 5フレームごとにログ出力
                        # if self.safety_mode:
                        #     self.get_logger().info(f'[SAFETY MODE] 手の姿勢を検出中（送信停止）: Frame {self.frame_count}')
                        # else:
                        #     self.get_logger().info(f'手の姿勢を検出・公開中: Frame {self.frame_count}')

                        self.get_logger().info(
                            f"Position: ({pose_msg.position.x:.3f}, {pose_msg.position.y:.3f}, {pose_msg.position.z:.3f})"
                        )
                        # self.get_logger().info(
                        #     f"Orientation: ({pose_msg.orientation.x:.3f}, {pose_msg.orientation.y:.3f}, "
                        #     f"{pose_msg.orientation.z:.3f}, {pose_msg.orientation.w:.3f})"
                        # )
                        # self.get_logger().info(
                        #     f"Hand Status: {hand_status} (Confidence: {confidence:.3f}) "
                        #     f"[Using Hands model]"
                        # )

                        # 深度推定の結果と処理時間を表示
                        if self.use_depth_estimation:
                            wrist_depth_str = f"{self.last_wrist_depth:.3f}" if self.last_wrist_depth is not None else 'N/A'
                            recorded_min_str = f"{self.recorded_depth_min:.3f}" if self.recorded_depth_min is not None else 'N/A'
                            recorded_max_str = f"{self.recorded_depth_max:.3f}" if self.recorded_depth_max is not None else 'N/A'

                            if self.last_corrected_x is not None and not self.use_plane_planning:
                                self.get_logger().info(
                                    f"Depth Correction: raw_depth={wrist_depth_str}, "
                                    f"original_x={self.last_original_x:.3f}m, corrected_x={self.last_corrected_x:.3f}m, "
                                    f"time={self.depth_estimation_time*1000:.1f}ms"
                                )
                            else:
                                self.get_logger().info(
                                    f"Depth Estimation: wrist_depth={wrist_depth_str}, "
                                    f"time={self.depth_estimation_time*1000:.1f}ms"
                                )

                            # 記録された最小値・最大値を表示
                            self.get_logger().info(
                                f"Recorded Depth Range: min={recorded_min_str}, max={recorded_max_str}"
                            )
                        self.get_logger().info(
                            f"Processing Time: MediaPipe={self.mediapipe_time*1000:.1f}ms, "
                            f"Total={self.video_processor.get_fps():.1f}fps"
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
            if hasattr(self, 'depth_estimator') and self.depth_estimator:
                self.depth_estimator.cleanup()
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