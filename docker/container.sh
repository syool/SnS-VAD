#!/bin/zsh

# run docker container
sudo nvidia-docker run -ti \
	-h shuso \
	--name shuso \
	--ipc host \
	-v $HOME/Github/shuso:/root/shuso:rw \
	-v $HOME/Documents/VADSET:/root/VADSET:ro \
	$USER/cuda:latest
