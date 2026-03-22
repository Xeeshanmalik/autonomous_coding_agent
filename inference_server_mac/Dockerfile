# ==========================================
# STAGE 1: The Hardened Builder 
# ==========================================
FROM registry.access.redhat.com/ubi9/ubi:latest AS builder

# FIX 1: Removed 'curl' to prevent the package conflict. (curl-minimal is already built-in)
RUN dnf install -y gcc-c++ make cmake git && \
    dnf clean all

WORKDIR /build

# Inject explicit memory-hardening flags (RELRO, SSP, PIE) during compilation
# Inject explicit memory-hardening flags for BOTH C and C++ compilers
ENV CFLAGS="-O3 -fstack-protector-strong -D_FORTIFY_SOURCE=2 -fPIE"
ENV CXXFLAGS="-O3 -fstack-protector-strong -D_FORTIFY_SOURCE=2 -fPIE"
ENV LDFLAGS="-Wl,-z,relro,-z,now -pie"

# Use CMAKE_INSTALL_PREFIX to cleanly package the binary AND its .so libraries
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git . && \
    cmake -B build -DGGML_NATIVE=OFF -DCMAKE_INSTALL_PREFIX=/opt/llama && \
    cmake --build build --config Release --target install -j$(nproc)

# Download the model (curl-minimal will handle this perfectly)
RUN curl -L -o /build/model.gguf "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf"

# ==========================================
# STAGE 2: Secure Runtime (Distroless-Style Zero Trust)
# ==========================================
FROM registry.access.redhat.com/ubi9/ubi-minimal:latest

# FIX 2: Added `libgomp` to provide the OpenMP multithreading library
RUN microdnf install -y libstdc++ libgcc libgomp && \
    microdnf clean all

WORKDIR /app

RUN echo "llmuser:x:10001:10001:LLM User:/app:/sbin/nologin" >> /etc/passwd && \
    echo "llmgroup:x:10001:" >> /etc/group

# Copy the perfectly preserved binaries and libraries from the builder
COPY --from=builder /opt/llama /usr/local/

# Explicitly copy the 4.5GB model from Stage 1 into the Stage 2 container
COPY --from=builder --chown=10001:10001 /build/model.gguf /models/model.gguf

# Tell the minimal OS where to find the newly copied shared libraries (libmtmd.so)
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:$LD_LIBRARY_PATH

# Enforce strict W^X immutability on the binary
RUN chown root:root /usr/local/bin/llama-server && \
    chmod 755 /usr/local/bin/llama-server

# Isolate stateful data directory permissions
RUN chown -R 10001:10001 /models

USER 10001:10001

EXPOSE 8080

# Binary path updated to reflect the clean CMake install
ENTRYPOINT ["/usr/local/bin/llama-server", "--host", "0.0.0.0", "--port", "8080", "-m", "/models/model.gguf", "-c", "16384"]