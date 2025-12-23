#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, Tuple, Union, Dict
from .config import HandPoseConfig
from .hand_detector import HandDetector
from utils import CvFpsCalc
from utils.download_file import download_file


class VideoProcessor:
    """
    カメラ/動画ファイルの処理とMediaPipe統合を行うクラス
    PoseとHandsの両モデルをサポート
    """
    
    def __init__(
        self,
        camera_device: Union[int, str] = 0,
        camera_width: int = 640,
        camera_height: int = 480,
        threshold_close_to_open: float = 0.35,
        threshold_open_to_close: float = 0.45,
        min_state_duration: float = 0.3
    ):
        """
        初期化

        Args:
        camera_device: カメラデバイス番号または動画ファイルパス
        camera_width: カメラ幅
        camera_height: カメラ高さ
        threshold_close_to_open: 閉→開の閾値
        threshold_open_to_close: 開→閉の閾値
        min_state_duration: 状態変化の最小持続時間（秒）
        """
        self.config = HandPoseConfig()

        # パラメータの設定
        self.camera_device = camera_device
        self.camera_width = camera_width
        self.camera_height = camera_height

        # ヒステリシス + 時間的連続性のパラメータ
        self.threshold_close_to_open = threshold_close_to_open
        self.threshold_open_to_close = threshold_open_to_close
        self.min_state_duration = min_state_duration

        # カメラとMediaPipeの初期化
        self.cap: Optional[cv2.VideoCapture] = None
        self.pose_detector: Optional[vision.PoseLandmarker] = None
        self.hand_detector: Optional[HandDetector] = None

        # FPS計測
        self.fps_calc = CvFpsCalc(buffer_len=10)

        # 初期化実行
        self._setup_camera()
        self._setup_mediapipe()

        # Handsモデルの初期化（常に有効）
        self.hand_detector = HandDetector(
            num_hands=2,
            threshold_close_to_open=self.threshold_close_to_open,
            threshold_open_to_close=self.threshold_open_to_close,
            min_state_duration=self.min_state_duration
        )

        print(f"VideoProcessor初期化完了:")
        print(f"  カメラデバイス: {self.camera_device}")
        print(f"  解像度: {self.camera_width}x{self.camera_height}")
        print(f"  Poseモデル: heavy（固定）")
        print(f"  Handsモデル: 有効（固定）")
        print(f"  閉→開閾値: {self.threshold_close_to_open}")
        print(f"  開→閉閾値: {self.threshold_open_to_close}")
        print(f"  最小持続時間: {self.min_state_duration}秒")

    def _setup_camera(self) -> None:
        """
        カメラ/動画ファイルの設定
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_device)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"カメラデバイス {self.camera_device} を開けませんでした")
            
            # カメラの解像度設定
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            
            # 実際の解像度を取得
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width != self.camera_width or actual_height != self.camera_height:
                print(f"警告: 要求解像度 {self.camera_width}x{self.camera_height} が設定できませんでした")
                print(f"実際の解像度: {actual_width}x{actual_height}")
                self.camera_width = actual_width
                self.camera_height = actual_height
            
            # バッファサイズを小さくして遅延を減らす
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print(f"カメラ設定完了: {actual_width}x{actual_height}")
            
        except Exception as e:
            print(f"カメラ設定エラー: {e}")
            raise

    def _setup_mediapipe(self) -> None:
        """
        MediaPipe PoseLandmarkerの設定（heavyモデル固定）
        """
        try:
            # モデルパスとURLを取得
            model_path = self.config.get_model_path()
            model_url = self.config.model_url

            # モデルディレクトリを作成
            os.makedirs(self.config.model_dir, exist_ok=True)

            # モデルファイルをダウンロード（存在しない場合）
            if not os.path.exists(model_path):
                print(f"Poseモデルファイルをダウンロード中: {model_url}")
                download_file(url=model_url, save_path=model_path)
                print(f"ダウンロード完了: {model_path}")

            # MediaPipe PoseLandmarkerを作成
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                min_pose_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence
            )

            self.pose_detector = vision.PoseLandmarker.create_from_options(options)
            print(f"MediaPipe PoseLandmarker作成完了")

        except Exception as e:
            print(f"MediaPipe設定エラー: {e}")
            raise

    def process_frame(self) -> Tuple[Optional[np.ndarray], Optional[vision.PoseLandmarkerResult], Optional[Dict]]:
        """
        フレームを取得し、姿勢検出と手の検出を実行
        ミラー表示で固定（推論前にフレームを反転）
        
        Returns:
            (フレーム画像, Pose検出結果, Hand検出結果) のタプル
            エラー時は (None, None, None)
        """
        if self.cap is None or self.pose_detector is None:
            return None, None, None
        
        try:
            # フレーム取得
            ret, frame = self.cap.read()
            if not ret:
                print("フレーム取得に失敗しました")
                return None, None, None
            
            # FPS計測更新
            self.fps_calc.get()
            
            # ミラー表示で固定（推論前にフレームを反転）
            frame = cv2.flip(frame, 1)
            
            # RGB変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Pose検出実行
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )
            pose_result = self.pose_detector.detect(mp_image)
            
            # Hand検出実行（常に有効）
            hand_result = None
            if self.hand_detector:
                hand_detection = self.hand_detector.detect_hands(rgb_frame)
                if hand_detection:
                    # 右手と左手を分離
                    right_hand, left_hand = self.hand_detector.separate_hands(hand_detection)
                    hand_result = {
                        'detection': hand_detection,
                        'right_hand': right_hand,
                        'left_hand': left_hand
                    }
            
            return frame, pose_result, hand_result
            
        except Exception as e:
            print(f"フレーム処理エラー: {e}")
            return None, None, None

    def get_fps(self) -> float:
        """
        現在のFPSを取得
        
        Returns:
            FPS値
        """
        return self.fps_calc.get()

    def get_frame_size(self) -> Tuple[int, int]:
        """
        フレームサイズを取得
        
        Returns:
            (幅, 高さ) のタプル
        """
        return self.camera_width, self.camera_height

    def is_camera_opened(self) -> bool:
        """
        カメラが開いているかチェック
        
        Returns:
            カメラの状態
        """
        return self.cap is not None and self.cap.isOpened()

    def set_camera_property(self, prop_id: int, value: float) -> bool:
        """
        カメラプロパティを設定
        
        Args:
            prop_id: OpenCVプロパティID
            value: 設定値
            
        Returns:
            設定成功フラグ
        """
        if self.cap is None:
            return False
        
        try:
            return self.cap.set(prop_id, value)
        except Exception as e:
            print(f"カメラプロパティ設定エラー: {e}")
            return False

    def get_camera_property(self, prop_id: int) -> float:
        """
        カメラプロパティを取得
        
        Args:
            prop_id: OpenCVプロパティID
            
        Returns:
            プロパティ値
        """
        if self.cap is None:
            return 0.0
        
        try:
            return self.cap.get(prop_id)
        except Exception as e:
            print(f"カメラプロパティ取得エラー: {e}")
            return 0.0

    def restart_camera(self) -> bool:
        """
        カメラを再起動
        
        Returns:
            再起動成功フラグ
        """
        try:
            # 現在のカメラを閉じる
            if self.cap is not None:
                self.cap.release()
            
            # カメラを再設定
            self._setup_camera()
            return True
            
        except Exception as e:
            print(f"カメラ再起動エラー: {e}")
            return False

    def cleanup(self) -> None:
        """
        リソースのクリーンアップ
        """
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                print("カメラリソースを解放しました")
            
            if self.hand_detector is not None:
                self.hand_detector.cleanup()
                self.hand_detector = None
                print("HandDetectorを解放しました")
                
        except Exception as e:
            print(f"クリーンアップエラー: {e}")

    def get_camera_info(self) -> dict:
        """
        カメラ情報を取得
        
        Returns:
            カメラ情報の辞書
        """
        if self.cap is None:
            return {}
        
        try:
            info = {
                'device': self.camera_device,
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'fourcc': int(self.cap.get(cv2.CAP_PROP_FOURCC)),
                'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
                'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
                'hue': self.cap.get(cv2.CAP_PROP_HUE),
                'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE),
                'auto_exposure': self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            }
            return info
        except Exception as e:
            print(f"カメラ情報取得エラー: {e}")
            return {}

    def __del__(self):
        """
        デストラクタ：リソースの自動クリーンアップ
        """
        self.cleanup()