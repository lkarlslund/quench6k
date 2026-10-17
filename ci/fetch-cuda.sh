#!/usr/bin/env bash
# Installs the CUDA compiler and runtime (no driver) from NVIDIA's redistributable archives.
#   ci/fetch-cuda.sh <version> <platform> <dir>     e.g. 13.4.3 linux-x86_64 /opt/cuda
# Platforms: linux-x86_64, linux-sbsa, windows-x86_64.
set -euo pipefail
version=$1 platform=$2 dir=$3
base=https://developer.download.nvidia.com/compute/cuda/redist
manifest=$(curl -fsSL "$base/redistrib_$version.json")
mkdir -p "$dir"
work=$(mktemp -d)
for pkg in cuda_nvcc cuda_cudart cuda_crt libnvvm cccl; do
    path=$(jq -r --arg p "$pkg" --arg plat "$platform" '.[$p][$plat].relative_path' <<<"$manifest")
    sha=$(jq -r --arg p "$pkg" --arg plat "$platform" '.[$p][$plat].sha256' <<<"$manifest")
    [ "$path" != null ] || { echo "no $pkg for $platform in CUDA $version" >&2; exit 1; }
    file="$work/$(basename "$path")"
    curl -fsSL -o "$file" "$base/$path"
    echo "$sha  $file" | sha256sum -c --quiet
    case "$file" in
        *.zip) unzip -q -o "$file" -d "$work/x" ;;
        *) mkdir -p "$work/x" && tar -xJf "$file" -C "$work/x" ;;
    esac
    cp -a "$work/x/$(basename "$path" | sed -E 's/\.(zip|tar\.xz)$//')"/. "$dir"/
    rm -rf "$work/x" "$file"
done
rm -rf "$work"
"$dir/bin/nvcc" --version | tail -1
