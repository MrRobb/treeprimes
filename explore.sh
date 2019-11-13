for seed in 'seq 0 100'
do
    cargo run --release -- -f consecutive -t 41 -g 100 -m 0.3 --min-depth 5 --max-depth 5 --min-value -3000 --max-value 3000 -p 10000 --seed $seed
done