hook := "--manifest-path nokozero_hook/Cargo.toml"

default: build

build: build-hook

build-hook:
    cargo build {{hook}} --release

fmt:
    cargo fmt --all {{hook}}

test-hook:
    cargo test {{hook}}

clippy-hook:
    cargo clippy {{hook}} --release --all-targets -- -D warnings

lint: clippy-hook
    cargo fmt --all {{hook}} --check

clean:
    cargo clean {{hook}}
