# Natural-Disaster-Image-Generation-to-raise-Environmental-Awareness

## Dataset Name and Source
## this dataset contains 5k image-text pairs
1. [multimodal-deep-learning-for-disaster-response](http://idl.iscram.org/files/husseinmouzannar/2018/2129_HusseinMouzannar_etal2018.pdf); [GITHUB](https://github.com/husseinmozannar/multimodal-deep-learning-for-disaster-response); [Dataset Download Link](https://drive.google.com/u/1/uc?id=1lLhTpfYBFaYwlAVaH7J-myHuN8mdV595&export=download)
## You can download our dataset here for Dalle-mini training:
1. If you want to finetune on the UCI-5k dataset, the encoded data is already available after cloning the repo; 
### If you want to finetune on our custom 40k dataset, you may need to download the data via this link: ;
```
cd dalle-mini-custom 
curl -L https://ucla.box.com/shared/static/szt6wcypjlqhj8d8885la5bd2jn50k8h --output aug_data.zip
unzip aug_data.zip -d data
```
### after downloading the raw dataset, we need to use our dataloader, and encoding model to encode them into an encoded data_dir for training
You may need to adjust GPU settings, the default is using all available gpus, and on a batch-size of 128
There will be a output dir created at path dalle-mini-custom/tools/src/encoded_output
```
cd dalle-mini-custom/tools/src
python encode_dataset_dallemini.py
```

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
### To train on 40k Custom dataset using default params
first, follow the dataset download and encoding step;
next, suppose with the encoded data at dalle-mini-custom/tools/src/encoded_output1
if you want to run validation along with training, plz split the .parquet files at encoded_output1 into two subfolders
train and validation, following the format of encoded_data dir.

```
cd dalle-mini-custom/tools/train

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup python train.py \
    --model_name_or_path dalle-mini/dalle-mini/model-1reghx5l:latest \
    --tokenizer_name dalle-mini/dalle-mini/model-1reghx5l:latest \
    --dataset_repo_or_path ../src/encoded_output1 \
    --warmup_steps 1\
    --streaming True \
    --learning_rate 0.00005\
    --num_train_epochs 3\
    --do_train True \
<!--     --do_eval True \ -->
    --output_dir ../subset_finetuned_model_lr5_adafactor &
```






## Evaluation

### FID Score
Compute FID score between two distributions
```
python src/fid_score.py /dir/with/images/from/distribution1 /dir/with/images/from/distribution2
```
- `--image-size` option (default 256) will center crop and resize images from both distributions to specified size
