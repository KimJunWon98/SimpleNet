gpus=(0 1)
datasets=(only-a only-b only-c only-d only-e only-f only-g only-h only-i only-j only-k)

for i in "${!datasets[@]}"; do
  d="${datasets[$i]}"
  gpu="${gpus[$(( i % ${#gpus[@]} ))]}"   # i를 GPU 개수로 나눈 나머지
  nohup bash ./run-wrn50-template.sh "$d" "$gpu" > "out-${d}.log" 2>&1 &
  echo "[LAUNCHED] $d on GPU $gpu (pid=$!)"
done
