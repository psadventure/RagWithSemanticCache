# LiteLLM gateway image with config.yaml baked in.
#
# The config is copied at build time instead of bind-mounted: a bind mount points at the
# host folder's inode, so if infra/config/ is ever replaced while the container runs (fresh
# clone, git checkout, sync tool, editor) the container silently sees an empty folder and the
# gateway serves no models. A copied file can't go stale.
#
# Apply config changes with:  docker compose up -d --build litellm   (from infra/)
# Secrets never enter the image: .dockerignore limits the build context to config/, and keys,
# the salt and gcloud ADC are passed at runtime by docker-compose.yml.

# Pinned to the release this repo was tested with (same as litellm in requirements.txt).
# To upgrade: change the tag, `docker compose up -d --build litellm`, run scripts/test_gateway.py.
FROM docker.litellm.ai/berriai/litellm:v1.109.1

COPY config/config.yaml /app/config/config.yaml
