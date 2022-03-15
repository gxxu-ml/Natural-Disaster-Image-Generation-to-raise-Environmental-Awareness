import jax
import jax.numpy as jnp
import os
# Load models & tokenizer
from dalle_mini.model import DalleBart, DalleBartTokenizer
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel
import wandb
from transformers import CLIPProcessor, CLIPModel
from dalle_mini.model import DalleBart, DalleBartTokenizer
from functools import partial
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange
from flax.training.common_utils import shard
import random
import torch
import pandas as pd
import random
from dalle_mini.text import TextNormalizer


###################### HYPERPARAMS TO SET


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
#whether to evaluate on the augmented validations set, if false, on 5k dataet.
VAL_ON_AUG = False
#whether to use the original pretrained model?
ORIGINAL = False

folders = [ "aug_finetuned_model1_lr2_adafactor",
           "aug_finetuned_model1_lr3_5k","aug_finetuned_model1_lr3_adafactor",
           "aug_finetuned_model1_lr4_5k","aug_finetuned_model1_lr4_adafactor",
           "aug_finetuned_model1_lr5_5k", "aug_finetuned_model1_lr5_adafactor"]
folders = ["ori"]
folders = ["subset_finetuned_model_lr3_adafactor","subset_finetuned_model_lr4_adafactor","subset_finetuned_model_lr5_adafactor"]


##################




# check how many devices are available
jax.local_device_count()

# type used for computation - use bfloat16 on TPU's
dtype = jnp.bfloat16 if jax.local_device_count() == 8 else jnp.float32

# TODO: fix issue with bfloat16
dtype = jnp.float32

DALLE_TOKENIZER = "dalle-mini/dalle-mini/model-1reghx5l:latest" 
DALLE_COMMIT_ID = None

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

# CLIP model
CLIP_REPO = "openai/clip-vit-base-patch16"
CLIP_COMMIT_ID = None

# number of predictions
n_predictions = 32

# We can customize top_k/top_p used for generating samples
gen_top_k = None
gen_top_p = None


    

# Model references
#set the model folder here; it determines model to load and output folder name

def load_model(DALLE_MODEL, config, ori):
    
    if ori:
        DALLE_MODEL = "dalle-mini/dalle-mini/model-1reghx5l:latest" 
        config = "../aug_finetuned_model1_lr5_adafactor/config.json"
    # Load models & tokenizer
    model = DalleBart.from_pretrained(
        DALLE_MODEL,
        config=config,
        #seed=training_args.seed_model,
        dtype=dtype,
        abstract_init=True,
        #load_on_cpu=True,
        # initializing params with gradient checkpointing creates issues
        # we correctly set it later per training_args
        gradient_checkpointing=False,
    )

    tokenizer = DalleBartTokenizer.from_pretrained(DALLE_TOKENIZER, revision=DALLE_COMMIT_ID)

    # Load VQGAN
    vqgan = VQModel.from_pretrained(VQGAN_REPO, revision=VQGAN_COMMIT_ID)

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, tokenizer, vqgan, clip, processor



# def generate(prompt,model,tokenizer,vqgan,clip,processor,text_normalizer,model_params,vqgan_params,p_generate,p_decode,key=key):

#     #takes the prompt, and generate image list and scores list: images and logits
#     processed_prompt = text_normalizer(prompt) if model.config.normalize_text else prompt
#     tokenized_prompt = tokenizer(
#         processed_prompt,
#         return_tensors="jax",
#         padding="max_length",
#         truncation=True,
#         max_length=128,
#     ).data
#     tokenized_prompt = replicate(tokenized_prompt)
    
    
#     # generate images
#     images = []
#     for i in trange(n_predictions // jax.device_count()):
#         # get a new key
#         key, subkey = jax.random.split(key)
#         # generate images
#         encoded_images = p_generate(
#             tokenized_prompt, shard_prng_key(subkey), model_params, gen_top_k, gen_top_p
#         )
#         # remove BOS
#         encoded_images = encoded_images.sequences[..., 1:]
#         #print("the length of the generated encoded image is: ",encoded_images.shape)
#         # decode images
#         decoded_images = p_decode(encoded_images, vqgan_params)
#         decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
#         #print("the shape of the decoded image is: ",decoded_images.shape)
#         for img in decoded_images:
#             images.append(Image.fromarray(np.asarray(img * 255, dtype=np.uint8)))
#         #print("the shape of the decoded image after post-processing is: ",len(images),images[0])
        
#     with torch.no_grad():
#         inputs = processor(text=[processed_prompt], images=images, 
#                            return_tensors="pt", padding='max_length',
#                            max_length=77, truncation=True)
#         outputs = clip(**inputs)
#         logits_per_image = outputs.logits_per_image
#         logits = logits_per_image.cpu().numpy().flatten()
#     return images, logits




def print_save(model_folder, val_on_aug=VAL_ON_AUG):

    DALLE_MODEL = os.path.join("..", model_folder, "flax_model.msgpack")
    
    config = os.path.join("..", model_folder, "config.json")

    #loading validation folder
    if val_on_aug:
        val_dir = "/home/gxu21/dalle-mini/cs269/data40k"
    else:
        val_dir = "/home/gxu21/dalle-mini/cs269/data"
    val_path = os.path.join(val_dir,"validation.txt")
    avg_max, avg_mean = 0,0
    df = pd.read_csv(val_path, sep=' ')
    #load model
    model, tokenizer, vqgan, clip, processor = load_model(DALLE_MODEL, config, ORIGINAL)
    
    # convert model parameters for inference if requested
    if dtype == jnp.bfloat16:
        model.params = model.to_bf16(model.params)

    model_params = replicate(model.params)
    vqgan_params = replicate(vqgan.params)
    #clip_params = replicate(clip.params)


    # model inference
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4))
    def p_generate(tokenized_prompt, key, params, top_k, top_p):
        return model.generate(
            **tokenized_prompt,
            do_sample=True,
            num_beams=1,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            max_length=257
        )

    # decode images
    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params):
        return vqgan.decode_code(indices, params=params)

    text_normalizer = TextNormalizer() if model.config.normalize_text else None

    for i in range(200):
        prompt_path = os.path.join(val_dir, df["captions"].iloc[i])
        with open(prompt_path) as f:
            prompt = f.readlines()[0]
        if val_on_aug:
            dir_path = os.path.join(model_folder,str(i))
        else:
            dir_path = os.path.join(model_folder+"_5kval",str(i))

        #generate images and logits
#         images,logits = generate(prompt=prompt,model=model, tokenizer=tokenizer, vqgan=vqgan, clip=clip, processor=processor,text_normalizer=text_normalizer,model_params=model_params,vqgan_params=vqgan_params,
#                                 p_generate=p_generate,p_decode=p_decode)
        #===========================================================================================
        # create a random key
        seed = random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)
        #takes the prompt, and generate image list and scores list: images and logits
        processed_prompt = text_normalizer(prompt) if model.config.normalize_text else prompt
        tokenized_prompt = tokenizer(
            processed_prompt,
            return_tensors="jax",
            padding="max_length",
            truncation=True,
            max_length=128,
        ).data
        tokenized_prompt = replicate(tokenized_prompt)


        # generate images
        images = []
        for i in trange(n_predictions // jax.device_count()):
            # get a new key
            key, subkey = jax.random.split(key)
            # generate images
            encoded_images = p_generate(
                tokenized_prompt, shard_prng_key(subkey), model_params, gen_top_k, gen_top_p
            )
            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]
            #print("the length of the generated encoded image is: ",encoded_images.shape)
            # decode images
            decoded_images = p_decode(encoded_images, vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            #print("the shape of the decoded image is: ",decoded_images.shape)
            for img in decoded_images:
                images.append(Image.fromarray(np.asarray(img * 255, dtype=np.uint8)))
            #print("the shape of the decoded image after post-processing is: ",len(images),images[0])

        with torch.no_grad():
            inputs = processor(text=[processed_prompt], images=images, 
                               return_tensors="pt", padding='max_length',
                               max_length=77, truncation=True)
            outputs = clip(**inputs)
            logits_per_image = outputs.logits_per_image
            logits = logits_per_image.cpu().numpy().flatten()

    
    
    
    #===========================================================================================
        #max_mean and avg_mean
        maxing = np.max(logits).item()
        mean = np.mean(logits).item()

        avg_max+=maxing
        avg_mean += mean

        # Check whether the specified path exists or not
        isExist = os.path.exists(dir_path)
        if not isExist:
          # Create a new directory because it does not exist 
            os.makedirs(dir_path)

        for idx,img in enumerate(images):
            score = logits[idx].item()
            img.save(os.path.join(dir_path, str(score)+"_"+str(idx)+".jpg"))

        text_file = open(os.path.join(dir_path,"prompt_content.txt"), "w")
        text_file.write(prompt + "\n")
        text_file.write("the max clip score is: " + str(max) +"\n")
        text_file.write("the mean clip score is: " + str(mean) +"\n")
        text_file.close()
        print(avg_max/float(i+1),avg_mean/float(i+1))

        
    if val_on_aug:
        text_file = open(os.path.join(model_folder,"final_scores.txt"), "w")
    else:
        text_file = open(os.path.join(model_folder+"_5kval","final_scores.txt"), "w")   
        
    text_file.write("the final average max score is:  "+ str(avg_max/200.0)+"\n")
    text_file.write("the final average mean score is:  "+ str(avg_mean/200.0)+"\n")
    text_file.close()  

    print("##################")
    print("the scoring for ", model_folder)
    print("the final average max score is:  ", avg_max/200.0)
    print("the final average mean score is:  ", avg_mean/200.0)

      

def main(folders=folders):
    for model_folder in folders:
        print_save(model_folder)
        
if __name__ == "__main__":
    main()









