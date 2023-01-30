# to build this dockerfile, run below:
# nvidia-docker build -t $USER/cuda:latest .

FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
MAINTAINER austinhyun

# install preliminaries
RUN apt-get update && apt upgrade -y
RUN apt-get install git curl zsh python3 python3-pip -y

# install PyTorch for Nvidia GeForce RTX30 series
RUN pip3 install torch==1.10.2+cu111 torchvision==0.11.3+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

# install dependencies
RUN pip3 install scikit-learn opencv-python \
    matplotlib plotly tqdm

# install oh-my-zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# install zsh theme "pure"
RUN mkdir -p "$HOME/.zsh-theme"
RUN git clone https://github.com/sindresorhus/pure.git "$HOME/.zsh-theme/pure"
RUN echo "fpath+=$HOME/.zsh-theme/pure\n \
          autoload -U promptinit; promptinit\n \
          zstyle :prompt:pure:git:stash show yes\n \
          zstyle :prompt:pure:path color '#40826d'\n \
          zstyle :prompt:pure:prompt:success color '#01DF01'\n \
          zstyle :prompt:pure:virtualenv color '#11a3a2'\n \
          zstyle :prompt:pure:git:branch color '#808080'\n \
          zstyle :prompt:pure:user color 'ffc100'\n \
          zstyle :prompt:pure:host color 'ffc100'\n \
          prompt pure" >> ~/.zshrc

WORKDIR /root

CMD [ "/bin/zsh" ]
