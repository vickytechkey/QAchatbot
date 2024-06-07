# QAchatbot

URL pattern http://127.0.0.1:8000/?q=who%20is%20vikram?

pass the query to llm model through varaiable q as provided in above

### To install docker in amazon linux , please follow below following command

1. sudo yum install docker -y  // to install docker package
2. sudo systemctl start docker // start docker file
3. sudo docker image ls // to check weather the docker is started or not ,  this command will display list of available docker images

To install git large file system use the following command
 sudo yum install git-lfs 

 To clone the model without large files GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/google/flan-t5-xxl

 command to install git-lfs in conda
 conda install conda-forge::git-lfs