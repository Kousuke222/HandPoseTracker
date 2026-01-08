#!/usr/bin/env python3
"""
Depth-Anything-V2 Webcam Real-time Demo
Webカメラからの映像をリアルタイムで深度推定するデモスクリプト

使用方法:
    python depth_anything_v2_webcam.py --encoder vits

キーボード操作:
    q: 終了
    c: 深度マップの色付けon/off
    f: フレームスキップon/off (処理を2フレームに1回実行)
    m: モデル切り替え (vits -> vitb -> vitl -> vits...)
    ESC: 終了
"""

import argparse
import cv2
import numpy as np
import torch
import time
from pathlib import Path

# Depth-Anything-V2のインポート
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    print("Error: Depth-Anything-V2がインストールされていません")
    print("インストール手順:")
    print("  git clone https://github.com/DepthAnything/Depth-Anything-V2.git")
    print("  cd Depth-Anything-V2")
    print("  pip install -r requirements.txt")
    exit(1)


class DepthEstimator:
    """深度推定クラス"""

    # モデル設定
    MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    def __init__(self, encoder='vits', device='cuda'):
        """
        初期化

        Args:
            encoder: モデルサイズ ('vits', 'vitb', 'vitl')
            device: 実行デバイス ('cuda' or 'cpu')
        """
        self.encoder = encoder
        self.device = device if torch.cuda.is_available() else 'cpu'

        if self.device == 'cpu' and device == 'cuda':
            print(f"Warning: CUDAが利用できません。CPUを使用します。")

        print(f"デバイス: {self.device}")
        print(f"モデル: {encoder}")

        # モデルの初期化
        self.model = None
        self.load_model(encoder)

    def load_model(self, encoder):
        """モデルをロード"""
        if encoder not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid encoder: {encoder}. Choose from {list(self.MODEL_CONFIGS.keys())}")

        config = self.MODEL_CONFIGS[encoder]

        print(f"Loading {encoder} model...")
        self.model = DepthAnythingV2(**config)

        # チェックポイントのパスを探す
        checkpoint_path = self.find_checkpoint(encoder)
        if checkpoint_path:
            print(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(f"Warning: チェックポイントが見つかりません。事前学習なしで実行します。")

        self.model.to(self.device)
        self.model.eval()
        self.encoder = encoder

    def find_checkpoint(self, encoder):
        """チェックポイントファイルを探す"""
        # 現在のスクリプトのディレクトリから相対的にパスを構築
        script_dir = Path(__file__).parent

        # よくあるパスのパターン
        possible_paths = [
            # DAV2_test -> hand_pose_tracker -> hand_pose_tracker -> src -> Depth-Anything-V2/checkpoints (3つ上)
            script_dir / "../../.." / "Depth-Anything-V2" / "checkpoints" / f"depth_anything_v2_{encoder}.pth",
            # カレントディレクトリからの相対パス
            Path(f"checkpoints/depth_anything_v2_{encoder}.pth"),
            Path(f"../checkpoints/depth_anything_v2_{encoder}.pth"),
            Path(f"../../Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth"),
            Path(f"../../../Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth"),
            # キャッシュディレクトリ
            Path.home() / f".cache/depth_anything_v2/depth_anything_v2_{encoder}.pth",
        ]

        for path in possible_paths:
            resolved_path = path.resolve()
            if resolved_path.exists():
                return str(resolved_path)
        return None

    def predict(self, image):
        """
        深度推定を実行

        Args:
            image: 入力画像 (numpy array, BGR format)

        Returns:
            depth: 深度マップ (numpy array)
        """
        # BGR -> RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 推論
        with torch.no_grad():
            depth = self.model.infer_image(rgb)

        return depth


class WebcamDemo:
    """Webカメラデモクラス"""

    def __init__(self, encoder='vits', camera_id=0, width=640, height=480):
        """
        初期化

        Args:
            encoder: モデルサイズ
            camera_id: カメラID
            width: キャプチャ幅
            height: キャプチャ高さ
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height

        # カメラの初期化
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(f"カメラ {camera_id} を開けませんでした")

        # 深度推定器の初期化
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.estimator = DepthEstimator(encoder=encoder, device=device)

        # 設定
        self.colorize = True
        self.frame_skip = False
        self.frame_count = 0

        # FPS計測
        self.fps = 0
        self.frame_times = []

        print("\n=== 操作方法 ===")
        print("q or ESC: 終了")
        print("c: 深度マップの色付けon/off")
        print("f: フレームスキップon/off")
        print("m: モデル切り替え (vits -> vitb -> vitl)")
        print("================\n")

    def colorize_depth(self, depth):
        """
        深度マップをカラー化

        Args:
            depth: 深度マップ

        Returns:
            カラー化された深度マップ
        """
        # 正規化
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        depth_normalized = (depth_normalized * 255).astype(np.uint8)

        # カラーマップを適用
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

        return depth_colored

    def update_fps(self, frame_time):
        """FPSを更新"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)

        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / sum(self.frame_times)

    def draw_info(self, image):
        """画像に情報を描画"""
        h, w = image.shape[:2]

        # 背景矩形
        cv2.rectangle(image, (10, 10), (320, 120), (0, 0, 0), -1)

        # テキスト
        y_offset = 30
        cv2.putText(image, f"FPS: {self.fps:.1f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(image, f"Model: {self.estimator.encoder}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(image, f"Colorize: {'ON' if self.colorize else 'OFF'}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(image, f"Frame Skip: {'ON' if self.frame_skip else 'OFF'}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return image

    def switch_model(self):
        """モデルを切り替え"""
        models = ['vits', 'vitb', 'vitl']
        current_idx = models.index(self.estimator.encoder)
        next_idx = (current_idx + 1) % len(models)
        next_model = models[next_idx]

        print(f"\nモデルを切り替え: {self.estimator.encoder} -> {next_model}")
        self.estimator.load_model(next_model)

    def run(self):
        """メインループを実行"""
        print("デモを開始します...")

        # 最後の深度マップをキャッシュ
        last_depth = None

        try:
            while True:
                start_time = time.time()

                # フレームをキャプチャ
                ret, frame = self.cap.read()
                if not ret:
                    print("フレームの取得に失敗しました")
                    break

                # フレームスキップ処理
                process_frame = True
                if self.frame_skip:
                    process_frame = (self.frame_count % 2 == 0)

                # 深度推定
                if process_frame:
                    depth = self.estimator.predict(frame)
                    last_depth = depth
                else:
                    depth = last_depth if last_depth is not None else np.zeros_like(frame[:, :, 0])

                # 深度マップの可視化
                if self.colorize:
                    depth_vis = self.colorize_depth(depth)
                else:
                    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
                    depth_normalized = (depth_normalized * 255).astype(np.uint8)
                    depth_vis = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)

                # 深度マップを入力画像と同じサイズにリサイズ
                depth_vis = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))

                # 情報を描画
                frame = self.draw_info(frame)
                depth_vis = self.draw_info(depth_vis)

                # 横に並べて表示
                combined = np.hstack([frame, depth_vis])

                cv2.imshow('Depth-Anything-V2 Demo (Left: Input, Right: Depth)', combined)

                # FPS更新
                frame_time = time.time() - start_time
                self.update_fps(frame_time)

                self.frame_count += 1

                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('c'):
                    self.colorize = not self.colorize
                    print(f"深度マップの色付け: {'ON' if self.colorize else 'OFF'}")
                elif key == ord('f'):
                    self.frame_skip = not self.frame_skip
                    print(f"フレームスキップ: {'ON' if self.frame_skip else 'OFF'}")
                elif key == ord('m'):
                    self.switch_model()

        except KeyboardInterrupt:
            print("\n中断されました")
        finally:
            self.cleanup()

    def cleanup(self):
        """リソースを解放"""
        print("\nクリーンアップ中...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("完了")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Depth-Anything-V2 Webcam Demo')
    parser.add_argument('--encoder', type=str, default='vits',
                       choices=['vits', 'vitb', 'vitl'],
                       help='モデルサイズ (default: vits)')
    parser.add_argument('--camera', type=int, default=0,
                       help='カメラID (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                       help='キャプチャ幅 (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='キャプチャ高さ (default: 480)')

    args = parser.parse_args()

    # デモを実行
    demo = WebcamDemo(
        encoder=args.encoder,
        camera_id=args.camera,
        width=args.width,
        height=args.height
    )
    demo.run()


if __name__ == '__main__':
    main()
