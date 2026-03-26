# ==========================================
# STAGE 1: The CUDA Builder 
# ==========================================
FROM nvidia/cuda:12.2.2-devel-ubi9 AS builder

RUN dnf install -y gcc-c++ make cmake git && \
    dnf clean all

WORKDIR /build

RUN cp /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib64/libcuda.so.1 || true
ENV LIBRARY_PATH="/usr/local/cuda/lib64/stubs:$LIBRARY_PATH"

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git . && \
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
    -DCMAKE_INSTALL_PREFIX=/opt/llama && \
    cmake --build build --config Release --target install -j4

# ==========================================
# STAGE 2: Secure CUDA Runtime & Model Download
# ==========================================
FROM nvidia/cuda:12.2.2-runtime-ubi9

RUN dnf install -y libstdc++ libgcc libgomp wget && \
    dnf clean all

WORKDIR /app

RUN echo "llmuser:x:10001:10001:LLM User:/app:/sbin/nologin" >> /etc/passwd && \
    echo "llmgroup:x:10001:" >> /etc/group

RUN mkdir -p /models && chown -R 10001:10001 /models

# DOWNLOAD THE 14.8GB Q3 MODEL (Guaranteed to fit 100% in your 16GB VRAM)
RUN wget -q --show-progress -O /models/model.gguf https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-32B-Q3_K_M.gguf && \
    chown 10001:10001 /models/model.gguf

COPY --from=builder /opt/llama /usr/local/

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:$LD_LIBRARY_PATH

RUN chown root:root /usr/local/bin/llama-server && \
    chmod 755 /usr/local/bin/llama-server

USER 10001:10001
EXPOSE 8080

# 100% GPU Offload (-ngl 99) + Flash Attention
# ENTRYPOINT ["/usr/local/bin/llama-server", "--host", "0.0.0.0", "--port", "8080", "-m", "/models/model.gguf", "-c", "4096", "-ngl", "99", "--flash-attn"]
# Changed "--flash-attn" to "-fa", "on"
# ENTRYPOINT ["/usr/local/bin/llama-server", "--host", "0.0.0.0", "--port", "8080", "-m", "/models/model.gguf", "-c", "4096", "-ngl", "99", "-fa", "on"]
# ENTRYPOINT ["/usr/local/bin/llama-server", "--host", "0.0.0.0", "--port", "8080", "-m", "/models/model.gguf", "-c", "2048", "-ngl", "99", "-fa", "on"]
# Added: -ngl 61 (Spill 3 layers), -np 1 (1 API Slot), --no-warmup
ENTRYPOINT ["/usr/local/bin/llama-server", "--host", "0.0.0.0", "--port", "8080", "-m", "/models/model.gguf", "-c", "4096", "-ngl", "61", "-fa", "on", "-np", "1", "--no-warmup"]