default: build

build: build-hook build-main

build-hook:
    cargo build -p nokozero_hook --target i686-pc-windows-gnu --release

build-main:
    cargo build -p nokozero --release

run *ARGS: build-hook
    cargo run -p nokozero --release -- {{ARGS}}

fmt:
    cargo fmt --all

clippy: clippy-hook clippy-main

clippy-hook:
    cargo clippy -p nokozero_hook --target i686-pc-windows-gnu --release --all-targets -- -D warnings

clippy-main:
    cargo clippy -p nokozero --release --all-targets -- -D warnings

clean:
    cargo clean
