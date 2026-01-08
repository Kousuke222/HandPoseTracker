#!/bin/bash
# Depth-Anything-V2 デモ実行スクリプト

# カレントディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# DAV2_test -> hand_pose_tracker -> hand_pose_tracker -> src (3つ上)
WORKSPACE_DIR="${SCRIPT_DIR}/../../.."

# Depth-Anything-V2のパスをPYTHONPATHに追加
DA_V2_PATH="${WORKSPACE_DIR}/Depth-Anything-V2"
export PYTHONPATH="${PYTHONPATH}:${DA_V2_PATH}"

echo "=========================================="
echo "Depth-Anything-V2 Webcam Demo"
echo "=========================================="
echo "Depth-Anything-V2 Path: ${DA_V2_PATH}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo ""

# デモを実行
cd "${SCRIPT_DIR}"
python3 depth_anything_v2_webcam.py "$@"
