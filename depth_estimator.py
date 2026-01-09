#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Optional, Tuple

# Depth-Anything-V2のインポート
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    print("Error: Depth-Anything-V2がインストールされていません")
    print("インストール手順:")
    print("  git clone https://github.com/DepthAnything/Depth-Anything-V2.git")
    print("  cd Depth-Anything-V2")
    print("  pip install -r requirements.txt")
    raise


class DepthEstimator:
    """
    Depth-Anything-V2を使用した深度推定クラス
    MediaPipeと並列で動作し、右手首の深度値を取得する
    """

    # モデル設定（vits固定）
    MODEL_CONFIG = {
        'encoder': 'vits',
        'features': 64,
        'out_channels': [48, 96, 192, 384]
    }

    def __init__(self, device: str = 'cuda'):
        """
        初期化

        Args:
            device: 実行デバイス ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'

        if self.device == 'cpu' and device == 'cuda':
            print(f"Warning: CUDAが利用できません。CPUを使用します。")

        print(f"DepthEstimator デバイス: {self.device}")
        print(f"DepthEstimator モデル: vits")

        # モデルの初期化
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """モデルをロード"""
        print(f"Loading vits model...")
        self.model = DepthAnythingV2(**self.MODEL_CONFIG)

        # チェックポイントのパスを探す
        checkpoint_path = self._find_checkpoint()
        if checkpoint_path:
            print(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(f"Warning: チェックポイントが見つかりません。事前学習なしで実行します。")

        self.model.to(self.device)
        self.model.eval()

    def _find_checkpoint(self) -> Optional[str]:
        """チェックポイントファイルを探す"""
        # 現在のスクリプトのディレクトリから相対的にパスを構築
        script_dir = Path(__file__).parent

        # # よくあるパスのパターン
        # possible_paths = [
        #     # hand_pose_tracker -> hand_pose_tracker -> src -> Depth-Anything-V2/checkpoints (2つ上)
        #     script_dir / "../.." / "Depth-Anything-V2" / "checkpoints" / "depth_anything_v2_vits.pth",
        #     # カレントディレクトリからの相対パス
        #     Path("checkpoints/depth_anything_v2_vits.pth"),
        #     Path("../checkpoints/depth_anything_v2_vits.pth"),
        #     Path("../../Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"),
        #     Path("../../../Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"),
        #     # キャッシュディレクトリ
        #     Path.home() / ".cache/depth_anything_v2/depth_anything_v2_vits.pth",
        # ]

        # for path in possible_paths:
        #     resolved_path = path.resolve()
        #     if resolved_path.exists():
        #         return str(resolved_path)
        # return None
        # '/home/xarm-in-case/xarm_ws/install/hand_pose_tracker/lib/python3.10
        # /site-packages/hand_pose_tracker
        # /../../Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth'

        return str(script_dir / "../../../../../../src/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth")

    def predict(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        深度推定を実行

        Args:
            image: 入力画像 (numpy array, BGR format)

        Returns:
            depth: 深度マップ (numpy array)、エラー時はNone
        """
        if self.model is None:
            return None

        try:
            # BGR -> RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 推論
            with torch.no_grad():
                depth = self.model.infer_image(rgb)

            return depth

        except Exception as e:
            print(f"深度推定エラー: {e}")
            return None

    def get_circular_depth(
        self,
        depth_map: np.ndarray,
        center_x: int,
        center_y: int,
        radius: int
    ) -> Optional[float]:
        """
        指定座標を中心とした円状領域の深度値の中央値を取得

        Args:
            depth_map: 深度マップ
            center_x: 中心X座標（画像座標系）
            center_y: 中心Y座標（画像座標系）
            radius: 円の半径（ピクセル）

        Returns:
            深度値の中央値、取得失敗時はNone
        """
        try:
            h, w = depth_map.shape

            # 範囲チェック
            if center_x < 0 or center_x >= w or center_y < 0 or center_y >= h:
                return None

            # 円状マスクの作成
            y_coords, x_coords = np.ogrid[:h, :w]
            mask = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2 <= radius ** 2

            # マスク領域の深度値を取得
            depth_values = depth_map[mask]

            if len(depth_values) == 0:
                return None

            # 中央値を計算
            median_depth = np.median(depth_values)

            return float(median_depth)

        except Exception as e:
            print(f"円状領域深度取得エラー: {e}")
            return None

    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("DepthEstimatorリソースを解放しました")
        except Exception as e:
            print(f"DepthEstimatorクリーンアップエラー: {e}")

    def __del__(self):
        """デストラクタ"""
        self.cleanup()
