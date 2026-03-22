# ==========================================
# STAGE 1: The Builder
# ==========================================
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies strictly from package-lock.json
COPY package*.json ./
RUN npm ci --ignore-scripts

# Copy the React source code and build it
COPY . .
RUN npm run build

# ==========================================
# STAGE 2: Secure Runtime (Zero CVE Target)
# ==========================================
FROM nginxinc/nginx-unprivileged:1.25-alpine-slim

# Copy the secure Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf


COPY --from=builder /app/dist /usr/share/nginx/html

# Explicitly switch to the non-root 'nginx' user (UID 101)
USER 101

# Expose the unprivileged port
EXPOSE 8080

# Sysdig and Kubernetes require Healthchecks to ensure container viability
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8080/ || exit 1

# Start Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]