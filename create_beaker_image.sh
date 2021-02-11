CURRENT=27
IMAGE_NAME=transformer_tools_v1
DOCKERFILE_NAME=Dockerfile

GIT_HASH=`git log --format="%h" -n 1`
IMAGE=$IMAGE_NAME_$USER-$GIT_HASH
IM_NAME=${IMAGE}_27

echo "Building $IMAGE"
docker build -f $DOCKERFILE_NAME -t $IMAGE .
beaker image create --name=${IM_NAME}_${CURRENT} --desc="TransformerToolsV${CURRENT}" $IMAGE
