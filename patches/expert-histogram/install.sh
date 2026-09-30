#!/bin/sh
# 把本目录复制进 llama.cpp 源码树的 examples/expert-histogram/，并注册子目录。
# 用法：sh install.sh /path/to/llama.cpp
set -e
SRC="${1:?用法: install.sh /path/to/llama.cpp}"
DST="$SRC/examples/expert-histogram"
mkdir -p "$DST"
cp expert-histogram.cpp CMakeLists.txt "$DST/"
LINE='    add_subdirectory(expert-histogram)'
F="$SRC/examples/CMakeLists.txt"
grep -qF "expert-histogram" "$F" || \
  sed -i "s#\(add_subdirectory(eval-callback)\)#\1\n$LINE#" "$F"
echo "已注册。构建："
echo "  cmake -B build -DLLAMA_CUBLAS=ON -DLLAMA_BUILD_EXAMPLES=ON"
echo "  cmake --build build --config Release -t llama-expert-histogram"
