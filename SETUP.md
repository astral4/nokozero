# Getting started

`nokozero` is currently designed for Linux. Here are instructions for getting started on Ubuntu Linux 24.04.

## Step 1: Update dependencies

Make sure all system dependencies are up to date.

```
sudo apt update
sudo apt upgrade
sudo apt autoremove
```

## Step 2: Install Wine

Follow the Wine [installation instructions](https://gitlab.winehq.org/wine/wine/-/wikis/Debian-Ubuntu) for Debian/Ubuntu.

```
# If your system is 64-bit, enable 32-bit architecture
sudo dpkg --add-architecture i386
# Download and add the Wine repository key
sudo mkdir -pm755 /etc/apt/keyrings
wget -O - https://dl.winehq.org/wine-builds/winehq.key | sudo gpg --dearmor -o /etc/apt/keyrings/winehq-archive.key -
# Add the Wine repository
sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/noble/winehq-noble.sources
# Install Wine Staging
sudo apt install --install-recommends winehq-staging
# Install additional 32-bit Wine dependencies
sudo apt install libgl1:i386 libvulkan1:i386
```

## Step 3: Install MinGW

`nokozero` uses MinGW to compile for Windows on 32-bit x86.

```
sudo apt install gcc-mingw-w64-i686
```

Note that this will install the following additional packages:

```
binutils-mingw-w64-i686 gcc-mingw-w64-base gcc-mingw-w64-i686-posix gcc-mingw-w64-i686-posix-runtime gcc-mingw-w64-i686-win32 gcc-mingw-w64-i686-win32-runtime mingw-w64-common mingw-w64-i686-dev
```

## Step 4: Install Rust

Follow the Rust [installation instructions](https://www.rust-lang.org/tools/install) for Unix-like operating systems.

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Optionally, you can add the Rust [compilation target](https://doc.rust-lang.org/stable/rustc/platform-support/windows-gnu.html) `i686-pc-windows-gnu` for Windows on 32-bit x86. Otherwise, Rust will automatically install components for `i686-pc-windows-gnu` when compiling the `nokozero_hook` crate.

```
rustup target add i686-pc-windows-gnu
```

## Step 5: Compile crates

`nokozero_hook` compiles to a 32-bit Windows library and intercepts the game's input reading for programmatic control. It must be compiled *before* compiling the main crate `nokozero`.

This repository comes with a [justfile](https://github.com/casey/just) for build orchestration. From the project root:

```
just build
```

Equivalently, without `just`:

```
cargo build --manifest-path nokozero_hook/Cargo.toml --release
```
