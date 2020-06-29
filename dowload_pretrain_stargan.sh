URL=https://www.dropbox.com/s/zdq6roqf63m0v5f/celeba-256x256-5attrs.zip?dl=0
ZIP_FILE=./stargan_celeba_256/models/celeba-256x256-5attrs.zip
mkdir -p ./stargan_celeba_256/models/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./stargan_celeba_256/models/
rm $ZIP_FILE