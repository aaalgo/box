# Setup
After cloning the repository, run the following.

Get Google's models.
```
cd box
git clone https://github.com/tensorflow/models
```

Build C++ code
```
python3 setup.py build
```

Get PicPac binary (only works for ubuntu 16.04)
```
wget www.aaalgo.com/picpac/binary/picpac.cpython-35m-x86_64-linux-gnu.so
```


# Data Preparation

Unzip to the `data` subdirectory or create a symbolic link, such that the following paths are valid
- ```box/lyft/data/Train/CameraRGB/1.png```
- ```box/lyft/data/Train/CameraSeg/1.png```


Create a directory or symbolic `box/cityscape/scratch`, which should be a directory backed by SSD storage.


# Importing

```
./import.py
```

Notes:
- We convert each object's mask into a polygon.
- We ignore self-car by default.
- We fill holes in objects.
- For each object we extract the longest contour (there exists many objects with multiple contours.)

# Training


