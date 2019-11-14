hyperfine "RAYON_NUM_THREADS={threads} cargo run --release -- -p 1000 -g 1000" -P threads 1 40 --export-csv "results/threads_sec.csv" --warmup 1
