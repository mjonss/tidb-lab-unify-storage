# Rebuild of the lab's VS Code image (cheesewang/tidb-lab-python-vscode:1.0), from its build history
# and the versions found inside it on 2026-10-30. The original is amd64 only; this also builds for arm64.
# Original extension versions: ms-python.python 2024.23.0-dev, ms-toolsai.jupyter 2024.11.0, eamodio.gitlens 16.2.1.
# The original uses Debian 11 (python:3.11.11-bullseye), but its security packages return 404 since Debian 11
# reached end of life (checked 2026-10-30), so this uses Debian 12 with the same Python. Node.js is 18 instead of 12.
FROM python:3.11.11-bookworm
RUN apt-get update && apt-get install -y curl wget git nodejs npm vim && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://code-server.dev/install.sh | sh -s -- --version 4.96.4
RUN code-server --install-extension ms-python.python \
    --install-extension ms-toolsai.jupyter \
    --install-extension eamodio.gitlens
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=2.0.1 python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry
WORKDIR /workspace
# No login, as in the lab image, so docker-compose.yml publishes its ports on localhost only.
RUN mkdir -p ~/.config/code-server && printf 'bind-addr: 0.0.0.0:8080\nauth: none\ncert: false\n' > ~/.config/code-server/config.yaml
EXPOSE 8000 8080
CMD ["code-server", "--host", "0.0.0.0", "--port", "8080"]
