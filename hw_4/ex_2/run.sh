lengths=(1048576 524288 262144)
segments=(131072 32768  8192)
trials=10

for trial in $(seq 1 $trials); do
  for len in ${lengths[@]}; do
    echo -n "$trial $len $len "
    ./streaming.out $len $len | grep kernel | cut -b 30-
    for seg in ${segments[@]}; do
      echo -n "$trial $len $seg "
      ./streaming.out $len $seg | grep kernel | cut -b 30-
    done;
  done;
done;
