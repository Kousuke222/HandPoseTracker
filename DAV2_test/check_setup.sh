#!/bin/bash
# Depth-Anything-V2 セットアップ確認スクリプト

echo "=========================================="
echo "Depth-Anything-V2 セットアップ確認"
echo "=========================================="
echo ""

# カレントディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="${SCRIPT_DIR}/../../.."
DA_V2_PATH="${WORKSPACE_DIR}/Depth-Anything-V2"

# チェック項目
check_count=0
pass_count=0

# 1. Depth-Anything-V2のディレクトリ確認
check_count=$((check_count + 1))
echo "[${check_count}] Depth-Anything-V2のディレクトリ確認..."
if [ -d "${DA_V2_PATH}" ]; then
    echo "  ✓ OK: ${DA_V2_PATH}"
    pass_count=$((pass_count + 1))
else
    echo "  ✗ NG: ディレクトリが見つかりません"
    echo "     -> ./setup_depth_anything_v2.sh を実行してください"
fi
echo ""

# 2. depth_anything_v2モジュール確認
check_count=$((check_count + 1))
echo "[${check_count}] depth_anything_v2モジュール確認..."
if [ -d "${DA_V2_PATH}/depth_anything_v2" ]; then
    echo "  ✓ OK: モジュールディレクトリが存在します"
    pass_count=$((pass_count + 1))
else
    echo "  ✗ NG: モジュールが見つかりません"
fi
echo ""

# 3. チェックポイントファイル確認
check_count=$((check_count + 1))
echo "[${check_count}] チェックポイントファイル確認..."
checkpoint_found=0
for model in vits vitb vitl; do
    checkpoint="${DA_V2_PATH}/checkpoints/depth_anything_v2_${model}.pth"
    if [ -f "${checkpoint}" ]; then
        size=$(du -h "${checkpoint}" | cut -f1)
        echo "  ✓ ${model}: ${size}"
        checkpoint_found=$((checkpoint_found + 1))
    fi
done

if [ ${checkpoint_found} -gt 0 ]; then
    echo "  ✓ OK: ${checkpoint_found}個のモデルが見つかりました"
    pass_count=$((pass_count + 1))
else
    echo "  ✗ NG: チェックポイントが見つかりません"
    echo "     -> ./setup_depth_anything_v2.sh を実行してください"
fi
echo ""

# 4. Python依存関係確認
check_count=$((check_count + 1))
echo "[${check_count}] Python依存関係確認..."
python3 -c "
import sys
missing = []
try:
    import torch
    print('  ✓ torch')
except ImportError:
    missing.append('torch')
    print('  ✗ torch')

try:
    import cv2
    print('  ✓ opencv-python')
except ImportError:
    missing.append('opencv-python')
    print('  ✗ opencv-python')

try:
    import numpy
    print('  ✓ numpy')
except ImportError:
    missing.append('numpy')
    print('  ✗ numpy')

if missing:
    print('')
    print('  不足しているパッケージ:', ', '.join(missing))
    print('  インストール: pip install --user', ' '.join(missing))
    sys.exit(1)
" 2>&1

if [ $? -eq 0 ]; then
    pass_count=$((pass_count + 1))
fi
echo ""

# 5. depth_anything_v2モジュールのインポート確認
check_count=$((check_count + 1))
echo "[${check_count}] depth_anything_v2モジュールのインポート確認..."
export PYTHONPATH="${PYTHONPATH}:${DA_V2_PATH}"
python3 -c "
try:
    from depth_anything_v2.dpt import DepthAnythingV2
    print('  ✓ OK: インポート成功')
except ImportError as e:
    print('  ✗ NG: インポート失敗')
    print(f'     エラー: {e}')
    exit(1)
" 2>&1

if [ $? -eq 0 ]; then
    pass_count=$((pass_count + 1))
fi
echo ""

# 結果表示
echo "=========================================="
echo "結果: ${pass_count}/${check_count} パス"
echo "=========================================="

if [ ${pass_count} -eq ${check_count} ]; then
    echo "✓ すべてのチェックに合格しました！"
    echo ""
    echo "次のコマンドでデモを実行できます:"
    echo "  ./run_demo.sh"
    exit 0
else
    echo "✗ いくつかの問題があります。上記のメッセージを確認してください。"
    exit 1
fi
