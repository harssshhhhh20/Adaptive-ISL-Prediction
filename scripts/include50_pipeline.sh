#!/bin/zsh
# INCLUDE-50 end to end: download -> landmarks -> 3 seeds x {fixed, adaptive}. Keeps the Mac awake.
cd "$(dirname $0)/.."
PY=/opt/anaconda3/envs/isl-cslr/bin/python
mkdir -p logs/include50
echo "[$(date +%T)] download"; $PY scripts/fetch_include50.py --workers 12 > logs/include50/fetch.log 2>&1
echo "[$(date +%T)] landmarks"; $PY scripts/extract_video_lm.py --videos data/include/videos --out features/include_lm.pkl --workers 4 > logs/include50/extract.log 2>&1
echo "[$(date +%T)] training"
for s in 0 1 2; do
  $PY demo/train_include50.py --fusion concat --seed $s > logs/include50/concat_s$s.log 2>&1 &
  $PY demo/train_include50.py --fusion trel_gate --seed $s > logs/include50/trel_gate_s$s.log 2>&1 &
  wait
done
echo "[$(date +%T)] done"; grep -h "TEST" logs/include50/*_s*.log
