# ==========================================
# STAGE 1: The CUDA Hardened Builder 
# ==========================================
# Swap from standard UBI9 to NVIDIA's UBI9 CUDA Devel image
FROM nvidia/cuda:12.2.2-devel-ubi9 AS builder

RUN dnf install -y gcc-c++ make cmake git && \
    dnf clean all

WORKDIR /build

# Maintain explicit memory-hardening flags
ENV CFLAGS="-O3 -fstack-protector-strong -D_FORTIFY_SOURCE=2 -fPIE"
ENV CXXFLAGS="-O3 -fstack-protector-strong -D_FORTIFY_SOURCE=2 -fPIE"
ENV LDFLAGS="-Wl,-z,relro,-z,now -pie"

# FIX: Change -DGGML_NATIVE=OFF to -DGGML_CUDA=ON
# This explicitly tells the C++ compiler to build the GPU tensor cores into the binary
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git . && \
    cmake -B build -DGGML_CUDA=ON -DCMAKE_INSTALL_PREFIX=/opt/llama && \
    cmake --build build --config Release --target install -j$(nproc)

# ==========================================
# STAGE 2: Secure CUDA Runtime
# ==========================================
# Swap to NVIDIA's UBI9 CUDA Runtime image (Distroless-Style)
FROM nvidia/cuda:12.2.2-runtime-ubi9

RUN microdnf install -y libstdc++ libgcc libgomp && \
    microdnf clean all

WORKDIR /app

RUN echo "llmuser:x:10001:10001:LLM User:/app:/sbin/nologin" >> /etc/passwd && \
    echo "llmgroup:x:10001:" >> /etc/group

# Copy the CUDA-compiled binaries and libraries
COPY --from=builder /opt/llama /usr/local/
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:$LD_LIBRARY_PATH

# Enforce strict W^X immutability
RUN chown root:root /usr/local/bin/llama-server && \
    chmod 755 /usr/local/bin/llama-server

# Isolate stateful data
RUN mkdir -p /models && chown -R 10001:10001 /models

USER 10001:10001
EXPOSE 8080

# FIX: Appended '-ngl 99' to the entrypoint. 
# This forces the server to offload all neural network layers directly into the RTX 3060's VRAM.
ENTRYPOINT ["/usr/local/bin/llama-server", "--host", "0.0.0.0", "--port", "8080", "-m", "/models/model.gguf", "-c", "16384", "-ngl", "99"]