# Depth-Anything-V2 Webカメラリアルタイム深度推定デモ

Depth-Anything-V2を使用してWebカメラからの映像をリアルタイムで深度推定するチュートリアルです。

## 目次

- [クイックスタート](#クイックスタート)
- [環境セットアップ](#環境セットアップ)
- [基本的な使い方](#基本的な使い方)
- [発展的な機能](#発展的な機能)
- [トラブルシューティング](#トラブルシューティング)

---

## クイックスタート

最速でデモを実行する手順:

```bash
# 1. このディレクトリに移動
cd ~/xarm_ws/src/hand_pose_tracker/hand_pose_tracker

# 2. 自動セットアップを実行
./setup_depth_anything_v2.sh

# 3. デモを実行
./run_demo.sh
```

デモが起動したら:
- 左側にWebカメラの映像、右側に深度マップが表示されます
- `c`キーで色付けON/OFF、`f`キーでフレームスキップON/OFF
- `m`キーでモデルを切り替え、`q`または`ESC`で終了

---

## 環境セットアップ

### 方法1: 自動セットアップ（推奨）

```bash
# セットアップスクリプトを実行
./setup_depth_anything_v2.sh
```

このスクリプトは以下を自動的に実行します:
- 依存関係のインストール
- Depth-Anything-V2のクローン
- モデル(Vit-Small)のダウンロード

### 方法2: 手動セットアップ

#### 1. 必要なライブラリのインストール

```bash
# 基本ライブラリのインストール
pip install torch torchvision opencv-python matplotlib numpy --user
```

#### 2. Depth-Anything-V2のリポジトリクローン

```bash
# ワークスペースのsrcディレクトリに移動
cd ~/xarm_ws/src

# リポジトリをクローン
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```

**注意**: Depth-Anything-V2には `setup.py` がないため、`pip install -e .` は**実行しないでください**。

#### 3. 事前学習済みモデルのダウンロード

```bash
# checkpointsディレクトリを作成
cd ~/xarm_ws/src/Depth-Anything-V2
mkdir -p checkpoints
cd checkpoints

# モデルをダウンロード (以下のいずれかまたは全て)

# Vit-Small (最軽量、最速) - 推奨
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

# Vit-Base (バランス型) - オプション
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth

# Vit-Large (最高精度) - オプション
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```

または、Hugging Faceから直接ダウンロード:
- [Vit-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small)
- [Vit-Base](https://huggingface.co/depth-anything/Depth-Anything-V2-Base)
- [Vit-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large)

#### 4. Pythonパスの設定

```bash
# Depth-Anything-V2をPythonパスに追加
export PYTHONPATH="${PYTHONPATH}:${HOME}/xarm_ws/src/Depth-Anything-V2"

# 永続化する場合は.bashrcまたは.zshrcに追加
echo 'export PYTHONPATH="${PYTHONPATH}:${HOME}/xarm_ws/src/Depth-Anything-V2"' >> ~/.bashrc
source ~/.bashrc
```

---

## 基本的な使い方

### 方法1: 実行スクリプトを使用（推奨）

```bash
# デフォルト設定で実行 (Vit-Smallモデル、カメラID=0)
./run_demo.sh

# パラメータを指定して実行
./run_demo.sh --encoder vitb
./run_demo.sh --encoder vits --camera 1 --width 1280 --height 720
```

このスクリプトは自動的にPythonパスを設定してデモを実行します。

### 方法2: Pythonパスを手動設定して実行

```bash
# Pythonパスを設定
export PYTHONPATH="${PYTHONPATH}:${HOME}/xarm_ws/src/Depth-Anything-V2"

# デフォルト設定で実行 (Vit-Smallモデル、カメラID=0)
python depth_anything_v2_webcam.py
```

### パラメータ指定

```bash
# Vit-Baseモデルで実行
./run_demo.sh --encoder vitb

# カメラIDを指定 (複数カメラがある場合)
./run_demo.sh --camera 1

# 解像度を指定
./run_demo.sh --width 1280 --height 720

# 全てのオプションを組み合わせ
./run_demo.sh --encoder vitl --camera 0 --width 1920 --height 1080
```

### コマンドライン引数

| 引数 | 説明 | デフォルト | 選択肢 |
|------|------|-----------|--------|
| `--encoder` | 使用するモデルサイズ | `vits` | `vits`, `vitb`, `vitl` |
| `--camera` | カメラID | `0` | 整数値 |
| `--width` | キャプチャ幅 | `640` | 整数値 |
| `--height` | キャプチャ高さ | `480` | 整数値 |

---

## 発展的な機能

デモ実行中に以下のキーボード操作が可能です。

### キーボード操作

| キー | 機能 | 説明 |
|------|------|------|
| `q` または `ESC` | 終了 | デモを終了します |
| `c` | 深度マップの色付けON/OFF | カラーマップ(INFERNO)とグレースケールを切り替え |
| `f` | フレームスキップON/OFF | 2フレームに1回処理してFPSを向上 |
| `m` | モデル切り替え | vits → vitb → vitl の順に循環 |

### 1. モデル選択機能

実行中に `m` キーを押すことで、以下の順でモデルを切り替えられます。

```
vits (軽量・高速) → vitb (バランス) → vitl (高精度) → vits ...
```

各モデルの特徴:

- **Vit-Small (vits)**: 最も軽量で高速。リアルタイム処理に最適
- **Vit-Base (vitb)**: 速度と精度のバランスが良い
- **Vit-Large (vitl)**: 最高精度だが処理が重い

### 2. フレームスキップ処理

`f` キーでON/OFFを切り替え。

- **OFF**: 全フレームを処理（高精度だが低FPS）
- **ON**: 2フレームに1回処理（FPSが約2倍に向上）

処理しないフレームでは前回の深度マップを表示します。

### 3. 深度マップの色付け

`c` キーでON/OFFを切り替え。

- **ON**: INFERNOカラーマップで可視化（暖色系）
- **OFF**: グレースケール表示

### 4. FPS表示

画面左上に以下の情報がリアルタイムで表示されます。

- **FPS**: 現在のフレームレート
- **Model**: 使用中のモデル
- **Colorize**: 色付けの状態
- **Frame Skip**: フレームスキップの状態

---

## 画面構成

デモ実行時は以下のように表示されます。

```
+---------------------------+---------------------------+
|     入力映像 (左)          |    深度マップ (右)          |
|  - Webカメラの映像         |  - 推定された深度情報       |
|  - FPS等の情報表示         |  - 近い:暗い / 遠い:明るい   |
+---------------------------+---------------------------+
```

---

## トラブルシューティング

### 1. pip install -e . でエラーが出る

**エラー**: `ERROR: neither 'setup.py' nor 'pyproject.toml' found.`

**原因**: Depth-Anything-V2リポジトリには `setup.py` がないため、editable installはサポートされていません。

**解決策**: `pip install -e .` は**実行しないでください**。代わりにPythonパスを設定します。
```bash
# Pythonパスに追加
export PYTHONPATH="${PYTHONPATH}:${HOME}/xarm_ws/src/Depth-Anything-V2"

# または、run_demo.shスクリプトを使用（自動的にパスを設定）
./run_demo.sh
```

### 2. モジュールが読み込めない

**エラー**: `ImportError: No module named 'depth_anything_v2'`

**解決策**:
```bash
# Pythonパスを設定
export PYTHONPATH="${PYTHONPATH}:${HOME}/xarm_ws/src/Depth-Anything-V2"

# 永続化する場合
echo 'export PYTHONPATH="${PYTHONPATH}:${HOME}/xarm_ws/src/Depth-Anything-V2"' >> ~/.bashrc
source ~/.bashrc

# または、run_demo.shスクリプトを使用
./run_demo.sh
```

### 3. チェックポイントが見つからない

**警告**: `Warning: チェックポイントが見つかりません`

**解決策**:
1. checkpointsディレクトリを作成: `mkdir -p ~/xarm_ws/src/Depth-Anything-V2/checkpoints`
2. モデルをダウンロード（上記の環境セットアップを参照）
3. ファイル名を確認: `depth_anything_v2_vits.pth` など

### 4. CUDAが利用できない

**警告**: `Warning: CUDAが利用できません。CPUを使用します。`

**解決策**:
```bash
# PyTorch (CUDA版) を再インストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA が正しくインストールされているか確認
python -c "import torch; print(torch.cuda.is_available())"
```

### 5. カメラが開けない

**エラー**: `RuntimeError: カメラ X を開けませんでした`

**解決策**:
```bash
# 利用可能なカメラを確認
ls /dev/video*

# 別のカメラIDを試す
python depth_anything_v2_webcam.py --camera 1
python depth_anything_v2_webcam.py --camera 2
```

### 6. FPSが低い

**問題**: フレームレートが5 FPS以下

**解決策**:
1. より軽量なモデルを使用: `--encoder vits`
2. フレームスキップをON: `f` キーを押す
3. 解像度を下げる: `--width 320 --height 240`
4. GPU (CUDA) を使用する

---

## パフォーマンス目安

参考までに、各モデルとハードウェアでの性能目安:

| モデル | GPU (RTX 3080) | GPU (GTX 1060) | CPU (i7-10700K) |
|--------|----------------|----------------|-----------------|
| vits   | ~60 FPS        | ~30 FPS        | ~5 FPS          |
| vitb   | ~40 FPS        | ~15 FPS        | ~2 FPS          |
| vitl   | ~25 FPS        | ~8 FPS         | ~1 FPS          |

※ 640x480解像度での目安

---

## コードの仕組み

### 主要クラス

#### 1. `DepthEstimator`

深度推定を担当するクラス。

```python
estimator = DepthEstimator(encoder='vits', device='cuda')
depth = estimator.predict(image)
```

主な機能:
- モデルのロードと管理
- チェックポイントの自動検索
- 深度推定の実行

#### 2. `WebcamDemo`

Webカメラデモを管理するクラス。

```python
demo = WebcamDemo(encoder='vits', camera_id=0)
demo.run()
```

主な機能:
- カメラキャプチャ
- リアルタイム処理
- UI表示とキー入力処理
- FPS計測

---

## 応用例

### カスタマイズ例1: 深度値の取得

```python
# 特定のピクセルの深度値を取得
depth = estimator.predict(frame)
center_depth = depth[frame.shape[0]//2, frame.shape[1]//2]
print(f"中心の深度: {center_depth:.2f}")
```

### カスタマイズ例2: 深度マップの保存

```python
# 深度マップを画像として保存
depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
depth_image = (depth_normalized * 255).astype(np.uint8)
cv2.imwrite('depth_map.png', depth_image)
```

### カスタマイズ例3: 動画ファイルへの適用

```python
# Webカメラの代わりに動画ファイルを使用
cap = cv2.VideoCapture('input_video.mp4')
# 以降は同じ処理
```

---

## 参考リンク

- [Depth-Anything-V2 公式リポジトリ](https://github.com/DepthAnything/Depth-Anything-V2)
- [論文](https://arxiv.org/abs/2406.09414)
- [Hugging Face](https://huggingface.co/depth-anything)

---

## ライセンス

このデモコードはMITライセンスの下で提供されます。
Depth-Anything-V2本体のライセンスについては、公式リポジトリを参照してください。
