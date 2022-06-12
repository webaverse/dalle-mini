### installation and set-up

# Model references

# dalle-mega
# DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
DALLE_MODEL = "./models/image"
DALLE_COMMIT_ID = None

# if the notebook crashes too often you can use dalle-mini instead by uncommenting below line
# DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"

# VQGAN model
# VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
# VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"
VQGAN_REPO = "./models/text"
VQGAN_COMMIT_ID = None

#

import jax
import jax.numpy as jnp

# check how many devices are available
jax.local_device_count()

#

# Load models & tokenizer
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel

# Load dalle-mini
model, params = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
)

# Load VQGAN
vqgan, vqgan_params = VQModel.from_pretrained(
    VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
)

#

from flax.jax_utils import replicate

params = replicate(params)
vqgan_params = replicate(vqgan_params)

#

from functools import partial

# model inference
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


# decode image
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)

#

import random

# create a random key
seed = random.randint(0, 2**32 - 1)
keys = jax.random.PRNGKey(seed)

### text prompt

from dalle_mini import DalleBartProcessor

processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

# server

from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange
import io
from flask import Flask, request, send_file, make_response, abort
# from random import randrange

app = Flask(__name__)

def mkResponse(data):
  return make_response(send_file(
    data,
    download_name="image.png",
    mimetype="image/png",
  ))

@app.route("/image")
def home():
    s = request.args.get("s")
    response = None
    if s is None or s == "":
        response = make_response("no text provided", 400)
    else:
        # prompts = ["witches and wizards anime screenshot"]
        prompts = [s]

        print("text 1")

        #

        print("text 2")
        tokenized_prompts = processor(prompts)

        #

        print("text 3")
        tokenized_prompt = replicate(tokenized_prompts)

        ### generate images

        print("generate 1")
        # number of predictions per prompt
        n_predictions = 1

        # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
        gen_top_k = None
        gen_top_p = None
        temperature = None
        cond_scale = 10.0

        #

        print("generate 2")


        print(f"Prompts: {prompts}\n")
        # generate images
        images = []
        for i in trange(max(n_predictions // jax.device_count(), 1)):
            print(f"Got 1 {i}\n")
            # get a new key
            key, subkey = jax.random.split(keys)
            print(f"Got 2 {i}\n")
            # generate images
            encoded_images = p_generate(
                tokenized_prompt,
                shard_prng_key(subkey),
                params,
                gen_top_k,
                gen_top_p,
                temperature,
                cond_scale,
            )
            print(f"Got 3 {i}\n")
            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]
            print(f"Got 4 {i}\n")
            # decode images
            decoded_images = p_decode(encoded_images, vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            print(f"Got 5 {i} {len(decoded_images)}\n")
            for j in range(len(decoded_images)):
                print(f"Got 6 {i} {j}\n")
                decoded_img = decoded_images[j]
                print(f"Got 7 {i}\n")
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                images.append(img)

                print(f"Got 8 {i}\n")
                # display(img)
                # print()
        print(f"Got 9 {i}\n")
        img = images[0]
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        # with open(f"out-{i}.png", "wb") as outfile:
        #     # Copy the BytesIO stream to the output file
        #     outfile.write(img_byte_arr.getbuffer())
        print(f"sending {len(img_byte_arr)} bytes...")
        response = mkResponse(img_byte_arr)
        
        print(f"Got A {i}\n")


    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    return response

def rank():
    ### rank images by CLIP score

    # CLIP model
    CLIP_REPO = "openai/clip-vit-base-patch32"
    CLIP_COMMIT_ID = None

    # Load CLIP
    clip, clip_params = FlaxCLIPModel.from_pretrained(
        CLIP_REPO, revision=CLIP_COMMIT_ID, dtype=jnp.float16, _do_init=False
    )
    clip_processor = CLIPProcessor.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)
    clip_params = replicate(clip_params)

    # score images
    @partial(jax.pmap, axis_name="batch")
    def p_clip(inputs, params):
        logits = clip(params=params, **inputs).logits_per_image
        return logits

    #

    from flax.training.common_utils import shard

    # get clip scores
    clip_inputs = clip_processor(
        text=prompts * jax.device_count(),
        images=images,
        return_tensors="np",
        padding="max_length",
        max_length=77,
        truncation=True,
    ).data
    logits = p_clip(shard(clip_inputs), clip_params)

    # organize scores per prompt
    p = len(prompts)
    logits = np.asarray([logits[:, i::p, i] for i in range(p)]).squeeze()
    #logits = rearrange(logits, '1 b p -> p b')

    #

    logits.shape

    #

    for i, prompt in enumerate(prompts):
        print(f"Prompt: {prompt}\n")
        for idx in logits[i].argsort()[::-1]:
            # display(images[idx*p+i])
            print(f"Score: {jnp.asarray(logits[i][idx], dtype=jnp.float32):.2f}\n")
        print()


if __name__ == "__main__":
    print("Starting server...")
    app.run(
        host="0.0.0.0",
        port=80,
        debug=False,
        # dev_tools_silence_routes_logging = False,
        # dev_tools_ui=True,
        # dev_tools_hot_reload=True,
        threaded=True,
    )