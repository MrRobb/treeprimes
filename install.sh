# Install Rust
curl https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env
cargo --version

# Build
cd treeprimes
export RUSTFLAGS="-C target-cpu=native"
cargo build --release

# Run
cd target/release/
./treeprimes --help