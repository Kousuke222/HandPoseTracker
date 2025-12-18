#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, Tuple, Dict, List
from utils.download_file import download_file


class HandDetector:
    """
    MediaPipe Handsモデルを使用した手の検出と開閉判定クラス
    """
    
    def __init__(self, num_hands: int = 2, model_dir: str = None):
        """
        初期化
        
        Args:
            num_hands: 検出する手の最大数
            model_dir: モデルファイルの保存ディレクトリ
        """
        self.num_hands = num_hands
        
        # モデル設定
        if model_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_dir = os.path.join(script_dir, '..', 'models')
        else:
            self.model_dir = model_dir
        
        # モデルURL
        self.model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
        
        # HandLandmarkerの初期化
        self.detector = None
        self._setup_hand_detector()
        
        # 手の開閉判定用の履歴
        self.hand_status_history = []
        self.history_size = 3
        
        print(f"HandDetector初期化完了:")
        print(f"  最大検出数: {self.num_hands}手")
        print(f"  モデルディレクトリ: {self.model_dir}")

    def _setup_hand_detector(self) -> None:
        """
        MediaPipe HandLandmarkerの設定
        """
        try:
            # モデルファイル名生成
            model_name = self.model_url.split('/')[-1]
            quantize_type = self.model_url.split('/')[-3]
            split_name = model_name.split('.')
            model_filename = f"{split_name[0]}_{quantize_type}.{split_name[1]}"
            model_path = os.path.join(self.model_dir, model_filename)
            
            # モデルディレクトリを作成
            os.makedirs(self.model_dir, exist_ok=True)
            
            # モデルファイルをダウンロード（存在しない場合）
            if not os.path.exists(model_path):
                print(f"Handモデルファイルをダウンロード中: {self.model_url}")
                download_file(url=self.model_url, save_path=model_path)
                print(f"ダウンロード完了: {model_path}")
            
            # HandLandmarker作成
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self.num_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.detector = vision.HandLandmarker.create_from_options(options)
            print(f"MediaPipe HandLandmarker作成完了")
            
        except Exception as e:
            print(f"HandLandmarker設定エラー: {e}")
            raise

    def detect_hands(self, rgb_frame: np.ndarray) -> Optional[vision.HandLandmarkerResult]:
        """
        手の検出を実行
        
        Args:
            rgb_frame: RGB形式の画像フレーム
            
        Returns:
            検出結果
        """
        if self.detector is None:
            return None
        
        try:
            # MediaPipe用の画像形式に変換
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )
            
            # 手の検出実行
            detection_result = self.detector.detect(mp_image)
            
            return detection_result
            
        except Exception as e:
            print(f"手検出エラー: {e}")
            return None

    def separate_hands(self, detection_result: vision.HandLandmarkerResult) -> Tuple[Dict, Dict]:
        """
        検出結果から右手と左手を分離
        ミラー表示のため、MediaPipeの'Left'が実際の右手、'Right'が実際の左手
        
        Args:
            detection_result: 手の検出結果
            
        Returns:
            Tuple[Dict, Dict]: (right_hand_data, left_hand_data)
        """
        right_hand_data = {
            'landmarks': None,
            'world_landmarks': None,
            'handedness': None,
            'index': None
        }
        
        left_hand_data = {
            'landmarks': None,
            'world_landmarks': None,
            'handedness': None,
            'index': None
        }
        
        if not detection_result.handedness:
            return right_hand_data, left_hand_data
        
        for i, handedness in enumerate(detection_result.handedness):
            hand_label = handedness[0].display_name
            
            # ミラー表示のため左右が逆転
            if hand_label == 'Left':  # MediaPipeの'Left'は実際の右手
                right_hand_data = {
                    'landmarks': detection_result.hand_landmarks[i] if i < len(detection_result.hand_landmarks) else None,
                    'world_landmarks': detection_result.hand_world_landmarks[i] if i < len(detection_result.hand_world_landmarks) else None,
                    'handedness': handedness,
                    'index': i
                }
            elif hand_label == 'Right':  # MediaPipeの'Right'は実際の左手
                left_hand_data = {
                    'landmarks': detection_result.hand_landmarks[i] if i < len(detection_result.hand_landmarks) else None,
                    'world_landmarks': detection_result.hand_world_landmarks[i] if i < len(detection_result.hand_world_landmarks) else None,
                    'handedness': handedness,
                    'index': i
                }
        
        return right_hand_data, left_hand_data

    def calculate_hand_openness(self, hand_landmarks, threshold: float = 0.4) -> Tuple[str, float]:
        """
        手の開閉状態を判定（添付コードと同等のアルゴリズム）
        
        Args:
            hand_landmarks: 手のランドマーク
            threshold: 開閉判定の閾値
            
        Returns:
            Tuple[str, float]: (状態('O'/'C'/'Unknown'), 信頼度(0-1))
        """
        if hand_landmarks is None:
            return "Unknown", 0.0
        
        try:
            # 各指のランドマークインデックス（MCP, PIP, DIP, TIP）
            fingers = {
                'thumb': [1, 2, 3, 4],      # 親指（CMC, MCP, IP, TIP）
                'index': [5, 6, 7, 8],      # 人差し指
                'middle': [9, 10, 11, 12],  # 中指
                'ring': [13, 14, 15, 16],   # 薬指
                'pinky': [17, 18, 19, 20]   # 小指
            }
            
            curl_ratios = []
            
            # 親指以外の4本の指で曲がり具合を計算
            for finger_name, indices in fingers.items():
                if finger_name == 'thumb':
                    # 親指は特別な処理（構造が異なるため）
                    continue
                    
                mcp, pip, dip, tip = [hand_landmarks[i] for i in indices]
                
                # 指の全長（MCP->TIP直線距離）
                full_length = np.sqrt(
                    (tip.x - mcp.x)**2 + (tip.y - mcp.y)**2 + (tip.z - mcp.z)**2
                )
                
                # 実際の関節を通る距離（MCP->PIP->DIP->TIP）
                actual_length = (
                    np.sqrt((pip.x - mcp.x)**2 + (pip.y - mcp.y)**2 + (pip.z - mcp.z)**2) +
                    np.sqrt((dip.x - pip.x)**2 + (dip.y - pip.y)**2 + (dip.z - pip.z)**2) +
                    np.sqrt((tip.x - dip.x)**2 + (tip.y - dip.y)**2 + (tip.z - dip.z)**2)
                )
                
                # 曲がり具合を計算（0: 伸びた状態, 1: 曲がった状態）
                if full_length > 0:
                    curl_ratio = min(1.0, max(0.0, (actual_length - full_length) / full_length))
                    curl_ratios.append(curl_ratio)
            
            # 親指の処理（簡易版）
            if 'thumb' in fingers:
                thumb_indices = fingers['thumb']
                cmc, mcp, ip, tip = [hand_landmarks[i] for i in thumb_indices]
                
                # 親指の曲がり具合（CMC->TIP）
                thumb_full = np.sqrt(
                    (tip.x - cmc.x)**2 + (tip.y - cmc.y)**2 + (tip.z - cmc.z)**2
                )
                thumb_actual = (
                    np.sqrt((mcp.x - cmc.x)**2 + (mcp.y - cmc.y)**2 + (mcp.z - cmc.z)**2) +
                    np.sqrt((ip.x - mcp.x)**2 + (ip.y - mcp.y)**2 + (ip.z - mcp.z)**2) +
                    np.sqrt((tip.x - ip.x)**2 + (tip.y - ip.y)**2 + (tip.z - ip.z)**2)
                )
                
                if thumb_full > 0:
                    thumb_curl = min(1.0, max(0.0, (thumb_actual - thumb_full) / thumb_full))
                    curl_ratios.append(thumb_curl * 0.8)  # 親指の重みを少し下げる
            
            if not curl_ratios:
                return "Unknown", 0.0
            
            # 平均曲がり具合
            avg_curl = np.mean(curl_ratios)
            
            # 信頼度計算（閾値から離れるほど信頼度が高い）
            confidence = 1.0 - abs(avg_curl - threshold) * 2
            confidence = min(1.0, max(0.0, confidence))
            
            # 閾値による判定
            status = "C" if avg_curl > threshold else "O"
            
            # 履歴による平滑化
            self.hand_status_history.append((status, confidence))
            if len(self.hand_status_history) > self.history_size:
                self.hand_status_history.pop(0)
            
            # 履歴から最頻値を選択
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
            print(f"手の開閉状態計算エラー: {e}")
            return "Unknown", 0.0

    def get_hand_landmarks_dict(self, hand_landmarks, image_width: int, image_height: int) -> Dict[int, List[int]]:
        """
        手のランドマークを辞書形式に変換
        
        Args:
            hand_landmarks: 手のランドマーク
            image_width: 画像幅
            image_height: 画像高さ
            
        Returns:
            ランドマーク辞書
        """
        landmark_dict = {}
        
        if hand_landmarks is None:
            return landmark_dict
        
        for index, landmark in enumerate(hand_landmarks):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_dict[index] = [landmark_x, landmark_y, landmark.z]
        
        return landmark_dict

    def get_world_landmarks_dict(self, hand_world_landmarks) -> Dict[int, List[float]]:
        """
        ワールド座標のランドマークを辞書形式に変換
        
        Args:
            hand_world_landmarks: ワールド座標のランドマーク
            
        Returns:
            ランドマーク辞書
        """
        landmark_dict = {}
        
        if hand_world_landmarks is None:
            return landmark_dict
        
        for index, landmark in enumerate(hand_world_landmarks):
            landmark_dict[index] = [landmark.x, landmark.y, landmark.z]
        
        return landmark_dict

    def cleanup(self) -> None:
        """
        リソースのクリーンアップ
        """
        self.detector = None
        print("HandDetectorリソースを解放しました")

    def __del__(self):
        """
        デストラクタ
        """
        self.cleanup()