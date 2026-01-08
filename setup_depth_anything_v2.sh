#!/bin/bash
# Depth-Anything-V2 セットアップスクリプト

set -e  # エラーが発生したら停止

echo "=========================================="
echo "Depth-Anything-V2 セットアップスクリプト"
echo "=========================================="
echo ""

# カレントディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="${SCRIPT_DIR}/../../.."

echo "ワークスペース: ${WORKSPACE_DIR}"
echo ""

# Step 1: 依存関係のインストール
echo "[1/4] 依存関係をインストール中..."
pip install torch torchvision opencv-python matplotlib numpy --user
echo "✓ 完了"
echo ""

# Step 2: Depth-Anything-V2のクローン（まだない場合）
cd "${WORKSPACE_DIR}"
if [ ! -d "Depth-Anything-V2" ]; then
    echo "[2/4] Depth-Anything-V2をクローン中..."
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git
    echo "✓ 完了"
else
    echo "[2/4] Depth-Anything-V2は既に存在します (スキップ)"
fi
echo ""

# Step 3: チェックポイントディレクトリの作成
echo "[3/4] チェックポイントディレクトリを作成中..."
cd "${WORKSPACE_DIR}/Depth-Anything-V2"
mkdir -p checkpoints
echo "✓ 完了"
echo ""

# Step 4: モデルのダウンロード
echo "[4/4] モデルをダウンロード中..."
cd checkpoints

# Vit-Small (最軽量)
if [ ! -f "depth_anything_v2_vits.pth" ]; then
    echo "  - Vit-Small (vits) をダウンロード中..."
    wget -q --show-progress https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
    echo "    ✓ 完了"
else
    echo "  - Vit-Small (vits) は既に存在します"
fi

# オプション: 他のモデルもダウンロードする場合はコメントアウトを外す
# # Vit-Base
# if [ ! -f "depth_anything_v2_vitb.pth" ]; then
#     echo "  - Vit-Base (vitb) をダウンロード中..."
#     wget -q --show-progress https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
#     echo "    ✓ 完了"
# else
#     echo "  - Vit-Base (vitb) は既に存在します"
# fi

# # Vit-Large
# if [ ! -f "depth_anything_v2_vitl.pth" ]; then
#     echo "  - Vit-Large (vitl) をダウンロード中..."
#     wget -q --show-progress https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
#     echo "    ✓ 完了"
# else
#     echo "  - Vit-Large (vitl) は既に存在します"
# fi

echo ""
echo "=========================================="
echo "セットアップ完了！"
echo "=========================================="
echo ""
echo "次のステップ:"
echo "1. Pythonパスを設定:"
echo "   export PYTHONPATH=\"\${PYTHONPATH}:${WORKSPACE_DIR}/Depth-Anything-V2\""
echo ""
echo "2. デモを実行:"
echo "   cd ${SCRIPT_DIR}"
echo "   python depth_anything_v2_webcam.py --encoder vits"
echo ""
echo "または、run_demo.sh を使用してください:"
echo "   cd ${SCRIPT_DIR}"
echo "   ./run_demo.sh"
echo ""
