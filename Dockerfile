# syntax=docker/dockerfile:1.7

# Base image for building
ARG LITELLM_BUILD_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:42df77a9974d6ec8b17a5ee8bc23b532600a44d705acef2409e0933c1251b45f

# Runtime image
ARG LITELLM_RUNTIME_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:42df77a9974d6ec8b17a5ee8bc23b532600a44d705acef2409e0933c1251b45f
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a
# Pinned by digest like the other base images; bump explicitly on Node upgrades.
ARG UI_BUILD_IMAGE=node:20.18-alpine3.20@sha256:3488b10bf958af7125a176419d2d8a9937d895bf124012aae811651988d2ffe6

FROM $UV_IMAGE AS uvbin

# Admin UI builder. Pinned to the build platform so the architecture-independent
# Next.js static export compiles once natively even in a multi-arch build,
# instead of once per target arch under QEMU.
FROM --platform=$BUILDPLATFORM $UI_BUILD_IMAGE AS ui-builder

ENV NEXT_TELEMETRY_DISABLED=1 \
    npm_config_fund=false \
    npm_config_audit=false

WORKDIR /ui

COPY ui/litellm-dashboard/package.json ui/litellm-dashboard/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --prefer-offline

COPY ui/litellm-dashboard/ ./
RUN npm run build

# Builder stage
FROM $LITELLM_BUILD_IMAGE AS builder

WORKDIR /app
USER root

COPY --from=uvbin /uv /usr/local/bin/uv
COPY --from=uvbin /uvx /usr/local/bin/uvx

# apk 从 apk.cgr.dev(Chainguard CDN)拉包；跨境/高延迟链路上单个包偶发
# HTTP 403/5xx,apk 会把它映射成 errno 打印成 "Permission denied"/"IO ERROR"
# 并整体失败。重试是幂等的:已装好的包会被跳过,只补拉失败的那一个。
RUN for attempt in 1 2 3 4 5; do \
        apk add --no-cache \
            bash \
            gcc \
            python3 \
            python3-dev \
            rust \
            openssl \
            openssl-dev \
            nodejs \
            npm \
            libsndfile \
        && exit 0; \
        echo "apk add failed (attempt $attempt/5), retrying in 5s..." >&2; \
        sleep 5; \
    done; \
    echo "apk add failed after 5 attempts" >&2; exit 1

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}"

# Optional PyPI file mirror. uv.lock pins an absolute
# https://files.pythonhosted.org/... URL per wheel/sdist, and `uv sync --frozen`
# does not re-resolve, so an index override (UV_DEFAULT_INDEX etc.) cannot
# redirect those downloads -- the URLs in the lock are what get fetched. From
# mainland China that means every package comes across the pacific one at a
# time; a cold build measured about two hours from the wulanchabu host.
#
# So rewrite the host in the lock at BUILD time instead. The lock stays
# pristine in git -- rewriting the committed file would collide with every
# upstream change to it, and this file is regenerated often upstream.
#
# This is safe because the rewrite touches ONLY the host: every sha256 in the
# lock is left alone, and uv verifies each download against it. A mirror
# serving different bytes fails the build loudly rather than silently
# installing something else. Verified the mirror uses the identical
# /packages/<hash-path>/<file> layout and returns a byte-identical size.
#
# Empty (the default) keeps upstream behaviour exactly.
ARG PYPI_FILES_MIRROR=""

# Optional crates.io mirror. The image installs rust because litellm-rust/
# builds a python-bridge extension, and `cargo metadata` has to fetch the
# crates.io index before it can resolve anything. Direct from mainland China
# that stalls hard -- observed at 19 minutes with the process alive at 0% CPU
# and no visible connection, which reads as a hang rather than slow progress.
# Same pattern the sandbox image already uses for its rust build.
# Empty (the default) keeps upstream behaviour.
ARG CARGO_REGISTRY_MIRROR=""
RUN if [ -n "$CARGO_REGISTRY_MIRROR" ]; then \
      mkdir -p "${CARGO_HOME:-/root/.cargo}"; \
      printf '[source.crates-io]\nreplace-with = "mirror"\n\n[source.mirror]\nregistry = "%s"\n' \
        "$CARGO_REGISTRY_MIRROR" > "${CARGO_HOME:-/root/.cargo}/config.toml"; \
    fi

# Copy dependency metadata first for layer caching
COPY pyproject.toml uv.lock ./
COPY enterprise/pyproject.toml enterprise/
COPY litellm-proxy-extras/pyproject.toml litellm-proxy-extras/

# Install third-party dependencies (cached unless pyproject.toml/uv.lock change)
RUN if [ -n "$PYPI_FILES_MIRROR" ]; then sed -i "s|https://files.pythonhosted.org/|${PYPI_FILES_MIRROR}|g" uv.lock; fi && \
    uv sync --frozen --no-install-project --no-install-workspace --no-default-groups --no-editable \
    --extra proxy \
    --extra proxy-runtime \
    --extra extra_proxy \
    --extra semantic-router \
    --python python3

# Copy full source tree
COPY . .

# Replace the committed UI bundle with the one built from this exact source.
# Clearing first drops the committed bundle's content-hashed chunks that COPY
# would otherwise leave behind alongside the fresh ones.
RUN rm -rf litellm/proxy/_experimental/out
COPY --from=ui-builder /ui/out/. litellm/proxy/_experimental/out/

# Build Admin UI before final sync (applies the enterprise color override when present)
RUN sed -i 's/\r$//' docker/build_admin_ui.sh && chmod +x docker/build_admin_ui.sh && ./docker/build_admin_ui.sh

# Install project and workspace packages (fast - deps already cached).
# The `COPY . .` above restored the pristine lock, so re-apply the rewrite.
RUN if [ -n "$PYPI_FILES_MIRROR" ]; then sed -i "s|https://files.pythonhosted.org/|${PYPI_FILES_MIRROR}|g" uv.lock; fi && \
    uv sync --frozen --no-default-groups --no-editable \
    --extra proxy \
    --extra proxy-runtime \
    --extra extra_proxy \
    --extra semantic-router \
    --python python3

RUN prisma generate --schema=./schema.prisma

RUN sed -i 's/\r$//' docker/entrypoint.sh && chmod +x docker/entrypoint.sh && \
    sed -i 's/\r$//' docker/prod_entrypoint.sh && chmod +x docker/prod_entrypoint.sh

# Runtime stage
FROM $LITELLM_RUNTIME_IMAGE AS runtime

USER root

# node (without npm) is required by the prisma CLI at runtime
RUN for attempt in 1 2 3 4 5; do \
        apk add --no-cache bash openssl tzdata nodejs python3 libsndfile && exit 0; \
        echo "apk add failed (attempt $attempt/5), retrying in 5s..." >&2; \
        sleep 5; \
    done; \
    echo "apk add failed after 5 attempts" >&2; exit 1

WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}"

# Copy only what runtime needs. The application is installed inside the venv;
# the rest of the builder's /app is source and build metadata that must not
# ship (manifest-scanning tools attribute everything in it to this image).
# entrypoint.sh invokes litellm/proxy/prisma_migration.py by source path.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/docker /app/docker
COPY --from=builder /app/schema.prisma /app/schema.prisma
COPY --from=builder /app/litellm/proxy/prisma_migration.py /app/litellm/proxy/prisma_migration.py
# enterprise/ is imported by source path at runtime (proxy_cli puts the
# working directory on sys.path; litellm/proxy/hooks resolves
# enterprise.enterprise_hooks from it)
COPY --from=builder /app/enterprise /app/enterprise
# Prisma binaries live in $HOME/.cache (default prisma-python location),
# which is /root/.cache here. Copy only the Prisma subdirs — copying the
# whole /root/.cache drags in the uv build cache (~660 MB, includes a
# setuptools wheel that surfaces as a CVE finding even though it's not
# on the runtime sys.path).
COPY --from=builder /root/.cache/prisma /root/.cache/prisma
COPY --from=builder /root/.cache/prisma-python /root/.cache/prisma-python

RUN find /app/.venv -type f -path "*/tornado/test/*" -delete && \
    find /app/.venv -type d -path "*/tornado/test" -delete

EXPOSE 4000/tcp

ENTRYPOINT ["docker/prod_entrypoint.sh"]
CMD ["--port", "4000"]
