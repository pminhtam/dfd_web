URL=https://www.dropbox.com/s/zdq6roqf63m0v5f/celeba-256x256-5attrs.zip?dl=0
ZIP_FILE=./stargan_celeba_256/models/celeba-256x256-5attrs.zip
mkdir -p ./stargan_celeba_256/models/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./stargan_celeba_256/models/
rm $ZIP_FILE

# StarGAN trained on CelebA (Black_Hair, Blond_Hair, Brown_Hair, Male, Young), 128x128 resolution
URL=https://www.dropbox.com/s/7e966qq0nlxwte4/celeba-128x128-5attrs.zip?dl=0
ZIP_FILE=./stargan_celeba_128/models/celeba-128x128-5attrs.zip
mkdir -p ./stargan_celeba_128/models/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./stargan_celeba_128/models/
rm $ZIP_FILE