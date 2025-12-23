#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import copy
import argparse
from typing import List, Any, Dict, Tuple, Union

import cv2
import numpy as np
import mediapipe as mp  
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision  
import matplotlib.pyplot as plt
from utils import CvFpsCalc
from utils.download_file import download_file

""" 
mediapipe奥行き補正アルゴリズム実装の雛形用コード

補正アルゴリズム概要：
- 補正を行うのは右肘(14)と右手首(16),右人差し指(20)のみ
- 入力
    - mediapipe Poseの検出結果(pose_world_landmarks)
    - カメラ焦点位置(World座標系,事前にキャリブレーションして得る)
    - 各関節の長さ(world座標系,事前にキャリブレーションして得る)
        - 右肩(12)~右肘(14)
        - 右肘(14)~右手首(16)
        - 右手首(16)~右人差し指(20)
- 出力
    - 補正済みの右肘(14)、右手首(16)、右人差し指(20)の座標で更新したpose_world_landmarks
- 手順
    1.事前準備:各関節の長さとカメラの焦点位置のキャリブレーション
        - 身体のキャリブレーションは肘の場所を明確化するため腕をL字に曲げて手を上げる姿勢で行う。
        - カメラの焦点位置の確定後は使用者は動かないものとする。
        1-1.待機状態(text:space to start calibration)
        1-2.space押下で3秒カウントダウン後、数十フレーム分のランドマーク検出を行う。(text:calibrating...)
        1-3.各フレームのランドマークから関節長さとカメラ焦点位置を推定し、中央値を採用。
        1-4.キャリブレーション状態から遷移し、補正を開始。
    2.中心が右肩、半径が右肩~右肘の長さの球を定義。(球の表面上のどこかに右肘が存在)
    3.カメラ焦点と右肘座標を通る直線を定義。(直線上のどこかに右肘が存在)
    4.球と直線の交点を計算し、２点を得る。
    5.補正前の右肘座標に近い交点を右肘座標として採用し、pose_world_landmarksを更新。 
        - 交点の採用判断:外れ値対策として前フレームとの連続性を考慮
    6.補正済みの右肘座標を使用して、同様の方法で右手首、右人差し指の座標も補正する。
- 追加機能
    - 補正前後のランドマークを3Dプロットで可視化
        - SHIFTキーのトグルで補正前のランドマーク表示/非表示切替
        - 同時に比較するため補正前後のランドマークを同一グラフに重ねて表示
    - 球と直線を3Dプロットで可視化
        - Ctrlキーのトグルで表示/非表示切替
    - spaceキーでキャリブレーションを再実行
- エッジケース(遭遇した場合、デバック用に事由と対応をPrint)
    - 球と直線が交わらない場合
        - 補正をスキップし、元の座標を使用する。
    - 球と直線が接する場合
        - 数値計算の誤差で2点が非常に近い場合も含む
        - 接する1点を採用
    - 2つの交点が補正前の座標から等距離
        - 前フレームの連続性から判断
    -  ランドマークの検出失敗
        - 補正をスキップし、元の座標を使用する。
    

        
"""

"""
mediapipe pose ランドマークインデックス：

0. nose
1. left_eye_inner
2. left_eye
3. left_eye_outer
4. right_eye_inner
5. right_eye
6. right_eye_outer
7. left_ear
8. right_ear
9. mouth_left
10. mouth_right
11. left_shoulder
12. right_shoulder
13. left_elbow
14. right_elbow
15. left_wrist
16. right_wrist
17. left_pinky
18. right_pinky
19. left_index
20. right_index
21. left_thumb
22. right_thumb
23. left_hip
24. right_hip
25. left_knee
26. right_knee
27. left_ankle
28. right_ankle
29. left_heel
30. right_heel
31. left_foot_index
32. right_foot_index
"""

# ランドマーク描画情報
POSE_LANDMARK_DRAW_INFO: Dict[int, Dict[str, Union[str, Tuple[int, int, int]]]] = {
    0: {'name': 'NOSE', 'color': (0, 255, 0)},
    1: {'name': 'LEFT_EYE_INNER', 'color': (255, 0, 0)},
    2: {'name': 'LEFT_EYE', 'color': (0, 0, 255)},
    3: {'name': 'LEFT_EYE_OUTER', 'color': (255, 255, 0)},
    4: {'name': 'RIGHT_EYE_INNER', 'color': (0, 255, 255)},
    5: {'name': 'RIGHT_EYE', 'color': (255, 0, 255)},
    6: {'name': 'RIGHT_EYE_OUTER', 'color': (128, 128, 128)},
    7: {'name': 'LEFT_EAR', 'color': (255, 128, 0)},
    8: {'name': 'RIGHT_EAR', 'color': (128, 0, 255)},
    9: {'name': 'MOUTH_LEFT', 'color': (0, 128, 255)},
    10: {'name': 'MOUTH_RIGHT', 'color': (128, 255, 0)},
    11: {'name': 'LEFT_SHOULDER', 'color': (255, 128, 128)},
    12: {'name': 'RIGHT_SHOULDER', 'color': (128, 128, 0)},
    13: {'name': 'LEFT_ELBOW', 'color': (0, 128, 128)},
    14: {'name': 'RIGHT_ELBOW', 'color': (128, 0, 128)},
    15: {'name': 'LEFT_WRIST', 'color': (64, 64, 64)},
    16: {'name': 'RIGHT_WRIST', 'color': (192, 192, 192)},
    17: {'name': 'LEFT_PINKY', 'color': (255, 69, 0)},
    18: {'name': 'RIGHT_PINKY', 'color': (75, 0, 130)},
    19: {'name': 'LEFT_INDEX', 'color': (173, 255, 47)},
    20: {'name': 'RIGHT_INDEX', 'color': (220, 20, 60)},
    21: {'name': 'LEFT_THUMB', 'color': (255, 0, 0)},
    22: {'name': 'RIGHT_THUMB', 'color': (0, 0, 255)},
    23: {'name': 'LEFT_HIP', 'color': (0, 255, 0)},
    24: {'name': 'RIGHT_HIP', 'color': (255, 255, 0)},
    25: {'name': 'LEFT_KNEE', 'color': (0, 255, 255)},
    26: {'name': 'RIGHT_KNEE', 'color': (255, 0, 255)},
    27: {'name': 'LEFT_ANKLE', 'color': (128, 128, 128)},
    28: {'name': 'RIGHT_ANKLE', 'color': (255, 128, 0)},
    29: {'name': 'LEFT_HEEL', 'color': (128, 0, 255)},
    30: {'name': 'RIGHT_HEEL', 'color': (0, 128, 255)},
    31: {'name': 'LEFT_FOOT_INDEX', 'color': (128, 255, 0)},
    32: {'name': 'RIGHT_FOOT_INDEX', 'color': (255, 128, 128)}
}

HAND_LANDMARK_DRAW_INFO: Dict[int, Dict[str, Union[str, Tuple[int, int, int]]]] = {
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
# 接続線情報
POSE_LINE_INFO_LIST: List[List[int]] = [
    [0, 1], [1, 2], [2, 3], [3, 7],  # 顔左側
    [0, 4], [4, 5], [5, 6], [6, 8],  # 顔右側
    [9, 10],  # 口
    [11, 12],  # 肩
    [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],  # 左腕
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22],  # 右腕
    [23, 24],  # 腰
    [23, 25], [25, 27], [27, 29], [29, 31],  # 左脚
    [24, 26], [26, 28], [28, 30], [30, 32],  # 右脚
    [11, 23], [12, 24]  # 胴体
]

HAND_LINE_INFO_LIST: List[List[int]] = [
    [0, 1], [1, 2], [2, 3], [3, 4],      # 親指
    [0, 5], [5, 6], [6, 7], [7, 8],      # 人差し指
    [0, 9], [9, 10], [10, 11], [11, 12], # 中指
    [0, 13], [13, 14], [14, 15], [15, 16], # 薬指
    [0, 17], [17, 18], [18, 19], [19, 20]  # 小指
]

# World座標プロット用インデックス
FACE_INDEX_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
RIGHT_ARM_INDEX_LIST = [11, 13, 15, 17, 19, 21]
LEFT_ARM_INDEX_LIST = [12, 14, 16, 18, 20, 22]
RIGHT_BODY_SIDE_INDEX_LIST = [11, 23, 25, 27, 29, 31]
LEFT_BODY_SIDE_INDEX_LIST = [12, 24, 26, 28, 30, 32]
SHOULDER_INDEX_LIST = [11, 12]
WAIST_INDEX_LIST = [23, 24]

PALM_INDEX_LIST = [0, 1, 5, 9, 13, 17, 0]
THUMB_INDEX_LIST = [1, 2, 3, 4]
INDEX_FINGER_INDEX_LIST = [5, 6, 7, 8]
MIDDLE_FINGER_INDEX_LIST = [9, 10, 11, 12]
RING_FINGER_INDEX_LIST = [13, 14, 15, 16]
PINKY_INDEX_LIST = [17, 18, 19, 20]

class CameraFocalPointEstimator:
    def __init__(self):
        self.camera_focal_point_world = None
        self.calibration_samples = []
        
    def estimate_camera_position(self, pose_world_landmarks, image_landmarks):
        """
        World Landmark座標系におけるカメラ位置を推定
        
        原理：
        - World空間の点P（例：右肩）
        - 画像上の点p（正規化座標、ミラー反転済み）
        - カメラ焦点Cから見て、pの方向にPが存在する
        
        座標系の定義：
        - World座標系: MediaPipe標準（右手系、腰中心が原点）
        - 描画座標系: X軸反転（ミラー表示に対応）
        - カメラはZ軸正方向（被写体の前方）に配置
        
        座標系の詳細:
        - MediaPipe World座標: 右手系、腰中心が原点、X軸が左→右
        - 本プログラムのWorld座標: X軸反転により左右反転 (X軸が右→左)
        - 画像座標: cv2.flip(frame, 1)により反転済み
        - これにより画像座標とWorld座標のX軸方向が一致
        """
        
        # 複数の関節点を使用して推定（精度向上）
        landmark_indices = [
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            mp.solutions.pose.PoseLandmark.NOSE,
        ]
        
        rays = []
        points = []
        
        for idx in landmark_indices:
            # World座標（X軸を反転して描画座標系に合わせる）
            wl = pose_world_landmarks[idx]
            world_point = np.array([-wl.x, wl.y, wl.z])
            
            # 画像座標（正規化、ミラー反転を考慮）
            il = image_landmarks[idx]
            
            # ミラー反転を考慮した正規化座標を[-1, 1]に変換
            # frame = cv2.flip(frame, 1)により、X座標が反転している
            img_x = (1.0 - il.x - 0.5) * 2  # ミラー反転: 1.0 - il.x
            img_y = (il.y - 0.5) * 2
            
            # カメラから見た方向ベクトル（Z軸正方向がカメラ視線方向）
            # ピンホールカメラモデル: 焦点距離を1.0と仮定
            ray_direction = np.array([img_x, img_y, 1.0])
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            
            rays.append(ray_direction)
            points.append(world_point)
        
        # 最小二乗法でカメラ位置を推定
        # P = C + t * d の形式で、Cを求める
        camera_position = self._solve_camera_position(rays, points)
        
        return camera_position
    
    def _solve_camera_position(self, rays, points):
        """
        複数の視線(ray)とWorld空間の点から、カメラ位置を推定
        
        各点について: point = camera_pos + t * ray
        最小二乗法で camera_pos を求める
        """
        n = len(rays)
        
        # 線形システムを構築
        # (I - ray * ray^T) * camera_pos = (I - ray * ray^T) * point
        A = np.zeros((3*n, 3))
        b = np.zeros(3*n)
        
        for i, (ray, point) in enumerate(zip(rays, points)):
            # 射影行列 (I - ray * ray^T)
            ray = ray.reshape(-1, 1)
            projection = np.eye(3) - ray @ ray.T
            
            A[3*i:3*i+3, :] = projection
            b[3*i:3*i+3] = projection @ point
        
        # 最小二乗解
        camera_pos, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        return camera_pos
    
    def calibrate(self, pose_world_landmarks, image_landmarks, num_samples=100):
        """
        複数フレームからカメラ位置を安定的に推定
        """
        camera_pos = self.estimate_camera_position(
            pose_world_landmarks, 
            image_landmarks
        )
        
        self.calibration_samples.append(camera_pos)
        
        # print("焦点計算を実行しました。残り",num_samples - len(self.calibration_samples),"回")
        
        if len(self.calibration_samples) >= num_samples:
            # 中央値を使用（外れ値に頑健）
            samples = np.array(self.calibration_samples)
            self.camera_focal_point_world = np.median(samples, axis=0)
            camera_focal_point_world : np.ndarray = self.camera_focal_point_world 
            self.camera_focal_point_world = np.array([
                camera_focal_point_world[0],  # X座標 (反転済み)
                camera_focal_point_world[1] + 0.25,  # Y座標(高さ),上方向に補正
                camera_focal_point_world[2],  # Z座標 (奥行き)
            ])
            print("キャリブレーション完了")
            print("カメラ焦点位置（World座標系）:", self.camera_focal_point_world)
            return True
        
        return False

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    #parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    #parser.add_argument("--height", help='cap height', type=int, default=480)
    parser.add_argument('--use_3d_plot', action='store_true', help='Use 3D plot for visualization')
    args = parser.parse_args()
    return args


def main() -> None:
    # 引数解析
    args = get_args()

    cap_device: Union[int, str] = args.device
    cap_width: int = args.width
    cap_height: int = args.height
    use_3d_plot: bool = True#一旦固定

    if args.video is not None:
        cap_device = args.video

    pose_model_url: str= 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task'
    #pose_model_url: str= 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
    hand_model_url: str= 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
    
    pose_model_path = model_downloader(pose_model_url)
    hand_model_path = model_downloader(hand_model_url)

    # カメラ準備
    cap: cv2.VideoCapture = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

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
    
    # FPS計測モジュール
    cvFpsCalc: CvFpsCalc = CvFpsCalc(buffer_len=10)

    # カメラ焦点推定モジュール
    camera_focal_point_estimator = CameraFocalPointEstimator()
    
    # World座標プロット準備
    if use_3d_plot :
        fig = plt.figure()
        ax = fig.add_subplot(211, projection="3d")
        r_ax = fig.add_subplot(212, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

    # 3秒待機(焦点計算後は動けないため)
    time.sleep(3)

    while True:
        display_fps: float = cvFpsCalc.get()

        # カメラキャプチャ
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            break
        
        # ===== 座標系の定義 =====
        # ミラー表示のため、フレーム全体をX軸反転
        # 以降の全ての座標系はこの反転後の座標系で統一
        # - 画像座標: 反転済みフレーム上の座標
        # - World座標: X軸を反転したMediaPipe World座標
        #   (world_x = -mediapipe_world_x, y,zはそのまま)
        frame = cv2.flip(frame, 1)  
        
        # 推論実施 
        rgb_frame: mp.Image = mp.Image(
            image_format=mp.ImageFormat.SRGBA,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA),
        )
        pose_detection_result: vision.PoseLandmarkerResult = pose_detector.detect(rgb_frame)#type: ignore
        hand_detection_result: vision.HandLandmarkerResult = hand_detector.detect(rgb_frame)#type: ignore

        # カメラ焦点位置推定
        if camera_focal_point_estimator.camera_focal_point_world is None :
            while True :
                calibrated: bool = camera_focal_point_estimator.calibrate(
                pose_detection_result.pose_world_landmarks[0],
                pose_detection_result.pose_landmarks[0],
                num_samples=30
                )
                if calibrated :
                    break
        
        camera_focal_point: np.ndarray = camera_focal_point_estimator.camera_focal_point_world

        # 描画
        debug_image: np.ndarray = copy.deepcopy(frame)
        debug_image = draw_pose_debug(
            debug_image,
            pose_detection_result,
        )
        
        debug_image = draw_hand_debug(
            debug_image,
            hand_detection_result,
        )

        # FPS描画
        debug_image = draw_fps(
            debug_image,
            display_fps
        )

        # 画面反映
        cv2.imshow('MediaPipe Demo3', debug_image)

        # 描画(ワールド座標)
        if use_3d_plot:
            draw_world_landmarks(
                plt,
                ax,
                pose_detection_result,
                camera_focal_point
            )
            # draw_hand_world_landmarks(
            #     plt,
            #     r_ax,
            #     hand_detection_result,
            # )

        # キー処理(ESC:終了)
        key: int = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

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

def create_landmark_dict(
    landmarks: Any,
    image_width: int,
    image_height: int,
) -> Dict[int, List[Union[int, float]]]:
    """ランドマーク情報を辞書形式に整理"""
    landmark_dict: Dict[int, List[Union[int, float]]] = {}
    visibility_threshold = 0.0 # ランドマークの可視性閾値
    presence_threshold = 0.0 # ランドマークの存在閾値
    
    for index, landmark in enumerate(landmarks):
        if landmark.visibility < visibility_threshold or landmark.presence < presence_threshold:
            continue
        landmark_x: int = min(int(landmark.x * image_width), image_width - 1)
        landmark_y: int = min(int(landmark.y * image_height), image_height - 1)
        landmark_dict[index] = [landmark_x, landmark_y, landmark.z]
    
    return landmark_dict

def draw_image_landmarks(
    image: np.ndarray,
    landmark_dict: Dict[int, List[Union[int, float]]],
    line_info_list: List[List[int]],
    landmark_draw_info: Dict[int, Dict[str, Union[str, Tuple[int, int, int]]]],
) -> np.ndarray:
    """ランドマークと接続線を描画（pose,hand統合版）
    
    Args:
        image: 描画対象の画像
        landmark_dict: ランドマーク辞書 {index: [x, y, z]}
        line_info_list: 接続線情報リスト（例：POSE_LINE_INFO_LIST）
        landmark_draw_info: ランドマーク描画情報辞書（例：POSE_LANDMARK_DRAW_INFO）
    
    Returns:
        描画済みの画像
    """
    # 接続線描画
    for line_info in line_info_list:
        if line_info[0] in landmark_dict and line_info[1] in landmark_dict:
            cv2.line(
                image,
                tuple(landmark_dict[line_info[0]][:2]), 
                tuple(landmark_dict[line_info[1]][:2]),  
                (220, 220, 220),  # 色
                2,  # 太さ
                cv2.LINE_AA  # 線の種類：アンチエイリアス
            )

    # 各ランドマーク描画
    for index, landmark in landmark_dict.items():
        cv2.circle(
            image,
            (landmark[0], landmark[1]),  
            3,  # 半径
            landmark_draw_info[index]['color'],  # 色
            -1,  # 塗りつぶし
            cv2.LINE_AA  # 線の種類：アンチエイリアス
        )
    
    return image

def draw_fps(
    image: np.ndarray,
    display_fps: float,
) -> np.ndarray:
    """FPS情報を描画"""
    cv2.putText(
        image,
        "FPS:" + str(display_fps),
        (10, 30),# 位置
        cv2.FONT_HERSHEY_SIMPLEX,# フォント
        1,# スケール
        (255, 255, 255),# 色
        2,# 太さ
        cv2.LINE_AA,# 線の種類：アンチエイリアス
    )
    return image

def draw_pose_debug(
    image: np.ndarray,
    detection_result: vision.PoseLandmarkerResult, # type: ignore
) -> np.ndarray:
    image_width, image_height = image.shape[1], image.shape[0]

    # ランドマーク描画
    for pose_landmarks in detection_result.pose_landmarks:
        # ランドマーク情報整理
        landmark_dict = create_landmark_dict(
            pose_landmarks,
            image_width,
            image_height
        )

        # ランドマークと接続線描画
        image = draw_image_landmarks(
            image, 
            landmark_dict, 
            POSE_LINE_INFO_LIST, 
            POSE_LANDMARK_DRAW_INFO
        )

    return image

def draw_hand_debug(
    image: np.ndarray,
    detection_result: vision.HandLandmarkerResult, # type: ignore
) -> np.ndarray:
    image_width, image_height = image.shape[1], image.shape[0]

    # 各手のランドマーク描画
    for handedness,hand_landmarks in zip(
            detection_result.handedness,
            detection_result.hand_landmarks,
    ):
        # ランドマーク情報整理
        landmark_dict = create_landmark_dict(
            hand_landmarks,
            image_width,
            image_height
        )

        # ランドマークと接続線描画
        image = draw_image_landmarks(
            image, 
            landmark_dict, 
            HAND_LINE_INFO_LIST, 
            HAND_LANDMARK_DRAW_INFO
        )

    return image

def extract_coordinates(
    landmark_dict: Dict[int, List[float]],
    index_list: List[int],
) -> Tuple[List[float], List[float], List[float]]:
    """指定されたインデックスリストから座標を抽出
    
    Args:
        landmark_dict: ランドマークインデックスと座標のマッピング
        index_list: 抽出したいランドマークのインデックスリスト
        
    Returns:
        x, y, z座標のリストのタプル
    """
    x_list, y_list, z_list = [], [], []
    for index in index_list:
        if index in landmark_dict:
            point = landmark_dict[index]
            x_list.append(point[0])
            y_list.append(point[2])
            z_list.append(point[1] * (-1))
    return x_list, y_list, z_list

def draw_world_landmarks(
    plt: Any,
    ax: Any,
    detection_result: vision.PoseLandmarkerResult, # type: ignore
    camera_focal_point: np.ndarray,
) -> None:
    """姿勢のWorld座標ランドマークを3D空間に描画
    
    Args:
        plt: matplotlibのpyplotオブジェクト
        ax: 3D軸オブジェクト
        detection_result: 姿勢検出結果
    """
    for pose_world_landmarks in detection_result.pose_world_landmarks:
        # ランドマーク情報整理
        landmark_dict: Dict[int, List[float]] = {}
        for index, landmark in enumerate(pose_world_landmarks):
            landmark_dict[index] = [-landmark.x, landmark.y, landmark.z]

        # 各部位の座標抽出
        face_x, face_y, face_z = extract_coordinates(landmark_dict, FACE_INDEX_LIST)
        right_arm_x, right_arm_y, right_arm_z = extract_coordinates(landmark_dict, RIGHT_ARM_INDEX_LIST)
        left_arm_x, left_arm_y, left_arm_z = extract_coordinates(landmark_dict, LEFT_ARM_INDEX_LIST)
        right_body_x, right_body_y, right_body_z = extract_coordinates(landmark_dict, RIGHT_BODY_SIDE_INDEX_LIST)
        left_body_x, left_body_y, left_body_z = extract_coordinates(landmark_dict, LEFT_BODY_SIDE_INDEX_LIST)
        shoulder_x, shoulder_y, shoulder_z = extract_coordinates(landmark_dict, SHOULDER_INDEX_LIST)
        waist_x, waist_y, waist_z = extract_coordinates(landmark_dict, WAIST_INDEX_LIST)

        # プロット
        ax.cla()
        ax.set_xlim3d(-2.0, 2.0)
        ax.set_ylim3d(-2.0, 2.0)
        ax.set_zlim3d(-2.0, 2.0)
        
        # 軸ラベル追加
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        
        # アスペクト比を立方体に設定
        ax.set_box_aspect([1, 1, 1])
        
        # グリッド表示
        ax.grid(True)

        # 色分けしてプロット
        ax.scatter(face_x, face_y, face_z, c='yellow', s=50, label='Face')
        ax.plot(right_arm_x, right_arm_y, right_arm_z, c='red', linewidth=2, label='Right Arm')
        ax.plot(left_arm_x, left_arm_y, left_arm_z, c='blue', linewidth=2, label='Left Arm')
        ax.plot(right_body_x, right_body_y, right_body_z, c='orange', linewidth=2)
        ax.plot(left_body_x, left_body_y, left_body_z, c='cyan', linewidth=2)
        ax.plot(shoulder_x, shoulder_y, shoulder_z, c='green', linewidth=2, label='Shoulder')
        ax.plot(waist_x, waist_y, waist_z, c='purple', linewidth=2, label='Waist')
        
        # カメラ焦点位置プロット
        ax.scatter(
            camera_focal_point[0],  # X座標（既に反転済み）
            camera_focal_point[2],  # Z座標（奥行き）
            camera_focal_point[1],  # Y座標（高さ、反転なし）
            c='black',
            s=100,
            marker='x',
            label='Camera Focal Point'
        )
        
        # 凡例表示(重複を避けるため一度だけ表示)
        ax.legend(loc='upper right')

    plt.pause(.001)

    return

def draw_hand_world_landmarks(
    plt: Any,
    ax: Any,
    detection_result: vision.HandLandmarkerResult, # type: ignore
) -> None:
    """手のWorld座標ランドマークを3D空間に描画(右手のみ)
    
    Args:
        plt: matplotlibのpyplotオブジェクト
        ax: 3D軸オブジェクト
        detection_result: 手検出結果
    """
    # 軸の初期化
    ax.cla()
    ax.set_xlim3d(-0.1, 0.1)
    ax.set_ylim3d(-0.1, 0.1)
    ax.set_zlim3d(-0.1, 0.1)
    
    # 軸ラベル追加
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    
    # アスペクト比を立方体に設定
    ax.set_box_aspect([1, 1, 1])
    
    # グリッド表示
    ax.grid(True)

    # 手が検出されない場合
    if not detection_result.hand_world_landmarks:
        plt.pause(.001)
        return

    for handedness, hand_world_landmarks in zip(
            detection_result.handedness,
            detection_result.hand_world_landmarks,
    ):
        # 左右判定(右手のみ処理)
        handedness_index: int = 0 if handedness[0].display_name == 'Left' else 1
        
        if handedness_index != 1:
            continue  # 右手以外はスキップ

        # ランドマーク情報整理
        landmark_dict: Dict[int, List[float]] = {}
        for index, landmark in enumerate(hand_world_landmarks):
            landmark_dict[index] = [landmark.x, landmark.y, landmark.z]

        # 各部位の座標抽出
        palm_x, palm_y, palm_z = extract_coordinates(landmark_dict, PALM_INDEX_LIST)
        thumb_x, thumb_y, thumb_z = extract_coordinates(landmark_dict, THUMB_INDEX_LIST)
        index_x, index_y, index_z = extract_coordinates(landmark_dict, INDEX_FINGER_INDEX_LIST)
        middle_x, middle_y, middle_z = extract_coordinates(landmark_dict, MIDDLE_FINGER_INDEX_LIST)
        ring_x, ring_y, ring_z = extract_coordinates(landmark_dict, RING_FINGER_INDEX_LIST)
        pinky_x, pinky_y, pinky_z = extract_coordinates(landmark_dict, PINKY_INDEX_LIST)

        # 色分けしてプロット
        ax.plot(palm_x, palm_y, palm_z, c='gray', linewidth=2, label='Palm')
        ax.plot(thumb_x, thumb_y, thumb_z, c='red', linewidth=2, label='Thumb')
        ax.plot(index_x, index_y, index_z, c='blue', linewidth=2, label='Index')
        ax.plot(middle_x, middle_y, middle_z, c='green', linewidth=2, label='Middle')
        ax.plot(ring_x, ring_y, ring_z, c='orange', linewidth=2, label='Ring')
        ax.plot(pinky_x, pinky_y, pinky_z, c='purple', linewidth=2, label='Pinky')
        
        # 凡例表示
        ax.legend(loc='upper right')

    plt.pause(.001)

    return

if __name__ == '__main__':
    main()
    
    