# Natural-Disaster-Image-Generation-to-raise-Environmental-Awareness

## Dataset Name and Source
## this dataset contains 5k image-text pairs
1. [multimodal-deep-learning-for-disaster-response](http://idl.iscram.org/files/husseinmouzannar/2018/2129_HusseinMouzannar_etal2018.pdf); [GITHUB](https://github.com/husseinmozannar/multimodal-deep-learning-for-disaster-response); [Dataset Download Link](https://drive.google.com/u/1/uc?id=1lLhTpfYBFaYwlAVaH7J-myHuN8mdV595&export=download)

## Evaluation

### FID Score
Compute FID score between two distributions
```
python src/fid_score.py /dir/with/images/from/distribution1 /dir/with/images/from/distribution2
```
- `--image-size` option (default 256) will center crop and resize images from both distributions to specified size
