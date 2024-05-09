epsilon=(0.05)
model=("mnist-10x2")
#traverse=("heuristic" "random")
traverse=("bounds")
for i in $(seq 0 1 99)
do
  for t in "${traverse[@]}"
  do
#    echo "$i" "$t"
    python mnist.py --index="$i" --traverse="$t"  --epsilon="$e" --network="$model"
  done
done





epsilon=(0.01)
#model=("gtsrb-10x2")
model=("gtsrb-cnn")
traverse=("bounds")
for i in $(seq 0 1 99)
do
  for t in "${traverse[@]}"
  do
    for e in "${epsilon[@]}"
    do
#      echo "$i" "$t" "$e"
      python gtsrb.py  --index="$i" --traverse="$t" --epsilon="$e" --network="$model"
    done
  done
done
