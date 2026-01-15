#!/bin/zsh
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1
pkill -f "update.py" >/dev/null 2>&1 || true
cleanup() {
    if [ -n "$PY_PID" ] && ps -p "$PY_PID" >/dev/null 2>&1; then
        kill "$PY_PID" >/dev/null 2>&1
    fi
    killall "TextEdit" >/dev/null 2>&1
    exit
}
trap cleanup SIGINT SIGTERM
open -a "TextEdit" "code design document.rtf" "updateCode.rtf"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.zsh hook)"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
fi
conda activate sim_env
python3 "update.py" &
PY_PID=$!
echo "Environment ready. Press Ctrl+O to close TextEdit and stop update.py."
while true; do
    read -sk 1 key || exit
    if [[ $key == $'\x0f' ]]; then
        cleanup
    fi
done