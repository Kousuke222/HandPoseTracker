#!/bin/bash
# Depth-Anything-V2 デモ実行スクリプト

# カレントディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="${SCRIPT_DIR}/../../.."

# Depth-Anything-V2のパスをPYTHONPATHに追加
export PYTHONPATH="${PYTHONPATH}:${WORKSPACE_DIR}/Depth-Anything-V2"

echo "=========================================="
echo "Depth-Anything-V2 Webcam Demo"
echo "=========================================="
echo "PYTHONPATH: ${PYTHONPATH}"
echo ""

# デモを実行
cd "${SCRIPT_DIR}"
python depth_anything_v2_webcam.py "$@"
