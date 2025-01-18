docker run \
  --gpus all \
  -ti \
  --rm \
  -v llava:/root/.cache \
  -v ./out:/app/out \
  -w /app \
  -u root \
  $(docker build . -q -t hunyuan-cli) \
  "$@"
