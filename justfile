hook := "--manifest-path nokozero_hook/Cargo.toml"

build-hook:
    cargo build {{hook}} --release

# `"$@"` rather than `{{ARGS}}`: an interpolation is re-split by the shell, so a game
# directory containing a space would reach argparse as two arguments.
[positional-arguments]
run *ARGS: build-hook
    uv run nokozero "$@"

fmt:
    cargo fmt --all {{hook}}
    uv run ruff format

test:
    cargo test {{hook}}
    uv run pytest

lint:
    cargo clippy {{hook}} --release --all-targets -- -D warnings
    cargo fmt --all {{hook}} --check
    uv run ruff check
    uv run ruff format --check
    uv run basedpyright

clean:
    cargo clean {{hook}}
