#/bin/bash -xe

VENV=.venv-docker
COMMAND=${1:-none}

docker_run()
{
    cmd=$1
    tag=${2:-tada:latest}
    docker run \
        --gpus all \
        --rm \
        -it \
        --shm-size=8g \
        -v `pwd`:/app \
        -v /mnt/d/dev/models/tada/data:/app/data \
        -v /mnt/d/dev/outputs/TADA:/app/output \
        -v /mnt/d/dev/.cache/huggingface/hub:/root/.cache/huggingface/hub \
        -w /app \
        $tag \
        $cmd
}


case $COMMAND in
    attach)
        docker_run bash "-it tada:latest"
        ;;
    setup)
        docker_run "/app/setup.sh $VENV"
        ;;
    build)
        docker build -t tada .
        ;;
    *)
        docker_run "/app/run.sh $VENV"
        ;;
esac

