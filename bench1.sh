hyperfine "RAYON_NUM_THREADS={threads} cargo run --release -- -p 10000 -g 1000" -P threads 1 100 -D 400 --export-csv "threads_sec.csv" --warmup 1