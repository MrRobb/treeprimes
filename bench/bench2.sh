for i in `seq 200 400 10000`
do
    hyperfine "cargo run --release -- -p {pop} -g $i" -P pop 200 10000 -D 400 --export-csv "results/gens_pop_$i.csv" --warmup 1
done