# Natural-Disaster-Image-Generation-to-raise-Environmental-Awareness

## Dataset Name and Source
## this dataset contains 5k image-text pairs
1. [multimodal-deep-learning-for-disaster-response](http://idl.iscram.org/files/husseinmouzannar/2018/2129_HusseinMouzannar_etal2018.pdf); [GITHUB](https://github.com/husseinmozannar/multimodal-deep-learning-for-disaster-response); [Dataset Download Link](https://drive.google.com/u/1/uc?id=1lLhTpfYBFaYwlAVaH7J-myHuN8mdV595&export=download)


## Training & Finetuning DALLE

### To directly train on 5k UCI dataset using default params
```
cd dalle-mini-custom/tools/train

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup python train.py \
    --model_name_or_path dalle-mini/dalle-mini/model-1reghx5l:latest \
    --tokenizer_name dalle-mini/dalle-mini/model-1reghx5l:latest \
    --dataset_repo_or_path ../src/encoded_data \
    --warmup_steps 1\
    --streaming True \
    --learning_rate 0.00005\
    --num_train_epochs 3\
    --do_train True \
    --do_eval True \
    --output_dir ../subset_finetuned_model_lr5_adafactor &
```





## Evaluation

### FID Score
Compute FID score between two distributions
```
python src/fid_score.py /dir/with/images/from/distribution1 /dir/with/images/from/distribution2
```
- `--image-size` option (default 256) will center crop and resize images from both distributions to specified size
