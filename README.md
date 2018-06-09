# OpenFaaS Style Transfer (Avatar-Net: Multi-scale Zero-shot Style Transfer by Feature Decoration. CVPR-2018)
* https://lucassheng.github.io/avatar-net/ 
* https://github.com/noido/avatar-net

# Check result
```
sudo docker run -it --net="host"  -e STYLE_PATH='styles/candy.jpg' -e INTER_WEIGHT=0.5 dfkozlov/openfaas-style-transfer
curl --request POST --url http://localhost:5000 --data-binary @"image.jpg" --output style_transfer_result.jpg
```

# Build docker image
* The trained model of Avatar-Net can be downloaded through the [Google Drive](https://drive.google.com/open?id=1_7x93xwZMhCL-kLrz4B2iZ01Y8Q7SlTX<Paste>).
* Pre-trained AvatarNet models, place them in the ./AvatarNet folder and create a file named "checkpoint" in the same directory (./AvatarNet) with the three .index, .data, .meta files. The file "checkpoint" (no extension) has the line model_checkpoint_path: "model.ckpt-120000". 
* The encoding layers of Avatar-Net are also borrowed from pretrained [VGG model](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz). Place the vgg_19 folder (with only vgg_19.ckpt inside) in the avatar-net/models folder.  
* In AvatarNet_config.yml, change the checkpoint_path on line 10 to point to the vgg_19.ckpt file. checkpoint_path: '../models/VGG/vgg_19.ckpt'
