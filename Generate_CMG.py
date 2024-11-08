import argparse
import itertools
import math
import os
import random

import shutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import accelerate
import argparse
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from slugify import slugify
from huggingface_hub import HfApi, HfFolder, CommitOperationAdd
from huggingface_hub import create_repo

from diffusers import DPMSolverMultistepScheduler
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#Your Generative Model Path
pretrained_model_name_or_path = "./models--stabilityai--stable-diffusion-2/snapshot/1e128c8891e52218b74cde8f26dbfc701cb99d79" 

new_label = {
    'whiptail, whiptail lizard': 'Cnemidophorus murinus',
    'nematode, nematode worm, roundworm': 'Nematoda',
    'loggerhead, loggerhead turtle, Caretta caretta': 'Caretta caretta',
    'scorpion': 'Scorpio',
    'vine snake': 'Oxybelis fulgidus',
    'hermit crab': 'Pagurus',
    'wallaby, brush kangaroo': 'Notamacropus',
    'tiger shark, Galeocerdo cuvieri': 'Galeocerdo cuvier',
    'axolotl, mud puppy, Ambystoma mexicanum': 'Ambystoma mexicanum',
    'sea snake': 'Hydrophiidae',
    'black swan, Cygnus atratus': 'Cygnus atratus',
    'stingray': 'Dasyatis',
    'thunder snake, worm snake, Carphophis amoenus': 'Carphophis amoenus',
    'pelican': 'Pelecanus',
    'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea': 'Dermochelys coriacea',
    'spotted salamander, Ambystoma maculatum': 'Ambystoma maculatum',
    'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias': 'Carcharodon carcharias',
    'crayfish, crawfish, crawdad, crawdaddy': 'Cambarus',
    'bald eagle, American eagle, Haliaeetus leucocephalus': 'Haliaeetus leucocephalus',
    'hen': 'Gallus gallus domesticus',
    'peacock': 'Pavo cristatus',
    'great grey owl, great gray owl, Strix nebulosa': 'Strix nebulosa',
    'wombat': 'Vombatus ursinus',
    'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita': 'Cacatua galerita',
    'mud turtle': 'Kinosternon',
    'spoonbill': 'Platalea',
    'hornbill': 'Buceros',
    'bittern': 'Botaurus',
    'night snake, Hypsiglena torquata': 'Hypsiglena torquata',
    'sea lion': 'Zalophus',
    'cock': 'Gallus gallus',
    'barn spider, Araneus cavaticus': 'Araneus cavaticus',
    'electric ray, crampfish, numbfish, torpedo': 'Torpedo',
    'red-backed sandpiper, dunlin, Erolia alpina': 'Calidris alpina',
    'chickadee': 'Poecile',
    'macaw': 'Ara',
    'common newt, Triturus vulgaris': 'Triturus vulgaris',
    'bulbul': 'Pycnonotus',
    'kite': 'Milvus',
    'tench, Tinca tinca': 'Tinca tinca',
    'prairie chicken, prairie grouse, prairie fowl': 'Tympanuchus',
    'hummingbird': 'Trochilidae',
    'rock crab, Cancer irroratus': 'Cancer irroratus',
    'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis': 'Varanus komodoensis',
    'black and gold garden spider, Argiope aurantia': 'Argiope aurantia',
    'harvestman, daddy longlegs, Phalangium opilio': 'Phalangium opilio',
    'chiton, coat-of-mail shell, sea cradle, polyplacophore': 'Polyplacophora',
    'limpkin, Aramus pictus': 'Aramus guarauna',
    'white stork, Ciconia ciconia': 'Ciconia ciconia',
    'sea slug, nudibranch': 'Nudibranchia',
    'goldfinch, Carduelis carduelis': 'Carduelis carduelis',
    'tick': 'Ixodida',
    'tmp': 'tmp',
    'sea anemone, anemone': 'Actiniaria',
    'king snake, kingsnake': 'Lampropeltis',
    'black grouse': 'Lyrurus tetrix',
    'garter snake, grass snake': 'Thamnophis',
    'toucan': 'Ramphastos',
    'wolf spider, hunting spider': 'Lycosidae',
    'goose': 'Anser',
    'drake': 'Anas platyrhynchos',
    'American alligator, Alligator mississipiensis': 'Alligator mississippiensis',
    'tarantula': 'Theraphosidae',
    'lorikeet': 'Trichoglossus',
    'banded gecko': 'Coleonyx variegatus',
    'jellyfish': 'Scyphozoa',
    'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish': 'Palinuridae',
    'bee eater': 'Meropidae',
    'bustard': 'Otididae',
    'magpie': 'Pica',
    'flatworm, platyhelminth': 'Platyhelminthes',
    'garden spider, Aranea diademata': 'Araneus diadematus',
    'coucal': 'Centropus',
    'hammerhead, hammerhead shark': 'Sphyrnidae',
    'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui': 'Ascaphus',
    'boa constrictor, Constrictor constrictor': 'Boa constrictor',
    'flamingo': 'Phoenicopterus',
    'terrapin': 'Malaclemys',
    'oystercatcher, oyster catcher': 'Haematopus',
    'albatross, mollymawk': 'Diomedeidae',
    'common iguana, iguana, Iguana iguana': 'Iguana iguana',
    'ptarmigan': 'Lagopus',
    'American coot, marsh hen, mud hen, water hen, Fulica americana': 'Fulica americana',
    'goldfish, Carassius auratus': 'Carassius auratus',
    'crane': 'Gruidae',
    'diamondback, diamondback rattlesnake, Crotalus adamanteus': 'Crotalus adamanteus',
    'green lizard, Lacerta viridis': 'Lacerta viridis',
    'conch': 'Strombus',
    'chambered nautilus, pearly nautilus, nautilus': 'Nautilus',
    'indigo bunting, indigo finch, indigo bird, Passerina cyanea': 'Passerina cyanea',
    'agama': 'Agama',
    'hognose snake, puff adder, sand viper': 'Heterodon',
    'water ouzel, dipper': 'Cinclus',
    'redshank, Tringa totanus': 'Tringa totanus',
    'sidewinder, horned rattlesnake, Crotalus cerastes': 'Crotalus cerastes',
    'Dungeness crab, Cancer magister': 'Metacarcinus magister',
    'green snake, grass snake': 'Opheodrys',
    'green mamba': 'Dendroaspis viridis',
    'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus': 'Cerastes cerastes',
    'black widow, Latrodectus mactans': 'Latrodectus mactans',
    'snail': 'Gastropoda'
}

origin_concept = {
    'whiptail, whiptail lizard': 'whiptail lizard',
    'nematode, nematode worm, roundworm': 'nematode worm',
    'loggerhead, loggerhead turtle, Caretta caretta': 'loggerhead turtle',
    'scorpion': 'scorpion',
    'vine snake': 'vine snake',
    'hermit crab': 'hermit crab',
    'wallaby, brush kangaroo': 'brush kangaroo',
    'tiger shark, Galeocerdo cuvieri': 'tiger shark',
    'axolotl, mud puppy, Ambystoma mexicanum': 'axolotl',
    'sea snake': 'sea snake',
    'black swan, Cygnus atratus': 'black swan',
    'stingray': 'stingray',
    'thunder snake, worm snake, Carphophis amoenus': 'worm snake',
    'pelican': 'pelican',
    'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea': 'leatherback turtle',
    'spotted salamander, Ambystoma maculatum': 'spotted salamander',
    'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias': 'great white shark',
    'crayfish, crawfish, crawdad, crawdaddy': 'crayfish',
    'bald eagle, American eagle, Haliaeetus leucocephalus': 'bald eagle',
    'hen': 'hen',
    'peacock': 'peacock',
    'great grey owl, great gray owl, Strix nebulosa': 'great grey owl',
    'wombat': 'wombat',
    'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita': 'sulphur-crested cockatoo',
    'mud turtle': 'mud turtle',
    'spoonbill': 'spoonbill',
    'hornbill': 'hornbill',
    'bittern': 'bittern',
    'night snake, Hypsiglena torquata': 'night snake',
    'sea lion': 'sea lion',
    'cock': 'cock',
    'barn spider, Araneus cavaticus': 'barn spider',
    'electric ray, crampfish, numbfish, torpedo': 'electric ray',
    'red-backed sandpiper, dunlin, Erolia alpina': 'red-backed sandpiper',
    'chickadee': 'chickadee',
    'macaw': 'macaw',
    'common newt, Triturus vulgaris': 'common newt',
    'bulbul': 'bulbul',
    'kite': 'kite',
    'tench, Tinca tinca': 'tench',
    'prairie chicken, prairie grouse, prairie fowl': 'prairie chicken',
    'hummingbird': 'hummingbird',
    'rock crab, Cancer irroratus': 'rock crab',
    'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis': 'Komodo dragon',
    'black and gold garden spider, Argiope aurantia': 'black and gold garden spider',
    'harvestman, daddy longlegs, Phalangium opilio': 'daddy longlegs',
    'chiton, coat-of-mail shell, sea cradle, polyplacophore': 'coat-of-mail shell',
    'limpkin, Aramus pictus': 'limpkin',
    'white stork, Ciconia ciconia': 'white stork',
    'sea slug, nudibranch': 'sea slug',
    'goldfinch, Carduelis carduelis': 'goldfinch',
    'tick': 'tick',
    'tmp': 'tmp',
    'sea anemone, anemone': 'sea anemone',
    'king snake, kingsnake': 'king snake',
    'black grouse': 'black grouse',
    'garter snake, grass snake': 'garter snake',
    'toucan': 'toucan',
    'wolf spider, hunting spider': 'wolf spider',
    'goose': 'goose',
    'drake': 'drake',
    'American alligator, Alligator mississipiensis': 'American alligator',
    'tarantula': 'tarantula',
    'lorikeet': 'lorikeet',
    'banded gecko': 'banded gecko',
    'jellyfish': 'jellyfish',
    'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish': 'spiny lobster',
    'bee eater': 'bee eater',
    'bustard': 'bustard',
    'magpie': 'magpie',
    'flatworm, platyhelminth': 'flatworm',
    'garden spider, Aranea diademata': 'garden spider',
    'coucal': 'coucal',
    'hammerhead, hammerhead shark': 'hammerhead shark',
    'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui': 'tailed frog',
    'boa constrictor, Constrictor constrictor': 'boa constrictor',
    'flamingo': 'flamingo',
    'terrapin': 'terrapin',
    'oystercatcher, oyster catcher': 'oystercatcher',
    'albatross, mollymawk': 'albatross',
    'common iguana, iguana, Iguana iguana': 'common iguana',
    'ptarmigan': 'ptarmigan',
    'American coot, marsh hen, mud hen, water hen, Fulica americana': 'American coot',
    'goldfish, Carassius auratus': 'goldfish',
    'crane': 'crane',
    'diamondback, diamondback rattlesnake, Crotalus adamanteus': 'diamondback rattlesnake',
    'green lizard, Lacerta viridis': 'green lizard',
    'conch': 'conch',
    'chambered nautilus, pearly nautilus, nautilus': 'chambered nautilus',
    'indigo bunting, indigo finch, indigo bird, Passerina cyanea': 'indigo bunting',
    'agama': 'agama',
    'hognose snake, puff adder, sand viper': 'hognose snake',
    'water ouzel, dipper': 'water ouzel',
    'redshank, Tringa totanus': 'redshank',
    'sidewinder, horned rattlesnake, Crotalus cerastes': 'sidewinder',
    'Dungeness crab, Cancer magister': 'Dungeness crab',
    'green snake, grass snake': 'green snake',
    'green mamba': 'green mamba',
    'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus': 'horned viper',
    'black widow, Latrodectus mactans': 'black widow',
    'snail': 'snail'
}



#@title Setup the prompt templates for training 
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

hyperparameters = {
        "learning_rate": 5e-04,
        "scale_lr": True,
        "max_train_steps": 500,
        "save_steps": 250,
        "train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "mixed_precision": "fp16",
        "seed": 42,
        "output_dir": "./sd-concept-output"
    }


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def create_dataloader(train_batch_size=1):
    return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def save_progress(text_encoder, placeholder_token_id, accelerator, save_path):
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)

def training_function(text_encoder, vae, unet):
    train_batch_size = hyperparameters["train_batch_size"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    learning_rate = hyperparameters["learning_rate"]
    max_train_steps = hyperparameters["max_train_steps"]
    output_dir = hyperparameters["output_dir"]
    gradient_checkpointing = hyperparameters["gradient_checkpointing"]

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=hyperparameters["mixed_precision"]
    )

    if gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    train_dataloader = create_dataloader(train_batch_size)

    if hyperparameters["scale_lr"]:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=learning_rate,
    )

    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.train()

    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states.to(weight_dtype)).sample

                 # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    grads = text_encoder.module.get_input_embeddings().weight.grad
                else:
                    grads = text_encoder.get_input_embeddings().weight.grad
                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % hyperparameters["save_steps"] == 0:
                    save_path = os.path.join(output_dir, f"learned_embeds-step-{global_step}.bin")
                    save_progress(text_encoder, placeholder_token_id, accelerator, save_path)

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()


    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)
        # Also save the newly trained embeddings
        save_path = os.path.join(output_dir, f"learned_embeds.bin")
        save_progress(text_encoder, placeholder_token_id, accelerator, save_path)
        
def getAllChildren(path):
    res=[]
    isChidren=1
    for file_path in os.listdir(path):
        if os.path.isdir(os.path.join(path,file_path)) and  'result' in file_path:
            isChidren=0
            break
        if os.path.isdir(os.path.join(path, file_path)):
            res+=getAllChildren(os.path.join(path, file_path))
            isChidren=0
        
    if isChidren:
        res+=[path]
    return res
        


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_path',type=str)
    parser.add_argument('-std',type=int)
    parser.add_argument('-domain',type=str)


    args = parser.parse_args()
    images_path = args.dataset_path #@param {type:"string"}
    std_cnt=int(args.std)
    domain=args.domain
    while not os.path.exists(str(images_path)):
        print('The images_path specified does not exist, use the colab file explorer to copy the path :')
        images_path=input("")
    data_paths=getAllChildren(images_path)
    print(data_paths)
    for save_path in data_paths: 
        if os.path.isdir(save_path+"/result"):
            shutil.rmtree(save_path+"/result")
        images = []
        cnt=0
        for file_path in os.listdir(save_path):
            try:
                rt=np.random.randint(0,10)
                cnt+=1
                image_path = os.path.join(save_path, file_path)
                images.append(Image.open(image_path).resize((512, 512)))
            except:
                print(f"{image_path} is not a valid image, please make sure to remove this file from the directory otherwise the training could fail.")
        image_grid(images, 1, len(images))
        print("Successfully load the images")
        print(len(images))
        oriconcept=save_path.split("/")[-3]
        # concept=concept.split(",")[-1]
        concept=new_label[oriconcept]
        concept="label_1"
        print(oriconcept)
        # domain=save_path.split("/")[-2]
        # if "_" in concept:
        #     concept=concept.split("_")[1]
        # print(domain," ",concept)
        domain=args.domain

        #@title Settings for your newly created concept
        #@markdown `what_to_teach`: what is it that you are teaching? `object` enables you to teach the model a new object to be used, `style` allows you to teach the model a new style one can use.
        what_to_teach = "object" #@param ["object", "style"]
        #@markdown `placeholder_token` is the token you are going to use to represent your new concept (so when you prompt the model, you will say "A `<my-placeholder-token>` in an amusement park"). We use angle brackets to differentiate a token from other words/tokens, to avoid collision.
        placeholder_token = "<"+domain+"-"+concept+">" 
        initializer_token = concept

        #@title Load the tokenizer and add the placeholder token as a additional special token.
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
        )

        # Add the placeholder token in tokenizer
        num_added_tokens = tokenizer.add_tokens(placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
        
        #@title Get token ids for our placeholder and initializer token. This code block will complain if initializer string is not a single token
        # Convert the initializer_token, placeholder_token to ids
        token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        #if len(token_ids) > 1:
            #print(initializer_token+": The initializer token must be a single token.")
            #continue
            #raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

        #@title Load the Stable Diffusion model
        # Load models and create wrapper for stable diffusion
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae"
        )
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet"
        )

        text_encoder.resize_token_embeddings(len(tokenizer))

        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

        # Freeze vae and unet
        freeze_params(vae.parameters())
        freeze_params(unet.parameters())
        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)


        train_dataset = TextualInversionDataset(
            data_root=save_path,
            tokenizer=tokenizer,
            size=vae.sample_size,
            placeholder_token=placeholder_token,
            repeats=100,
            learnable_property=what_to_teach, #Option selected above between object and style
            center_crop=False,
            set="train",
        )

        noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")
        rescnt=0
        if not os.path.isdir(save_path+"/result"):
            os.makedirs(save_path+"/result")
            save_path+="/result"
        else:
            shutil.rmtree(save_path+"/result")
            # while os.path.isdir(save_path+"/result"):
            #     rescnt+=1
            # os.makedirs(save_path+"/result"+str(rescnt))
            # save_path+="/result"+str(rescnt)


        logger = get_logger(__name__)

        accelerate.notebook_launcher(training_function, args=(text_encoder, vae, unet),num_processes=1)

        for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
            if param.grad is not None:
                del param.grad  # free some memory
            torch.cuda.empty_cache()

        #@title Set up the pipeline 
        
        pipe = StableDiffusionPipeline.from_pretrained(
            hyperparameters["output_dir"],
            scheduler=DPMSolverMultistepScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler"),
            torch_dtype=torch.float16,
            local_files_only = True
        ).to("cuda")

        #@title Run the Stable Diffusion pipeline
        #@markdown Don't forget to use the placeholder token in your prompt

        #prompt = "a <cat-toy> inside ramen-bowl" #@param {type:"string"}
        food_scenarios = [
            "On a picnic blanket",
            "On a table",
            "In a trash bin",
            "Inside a refrigerator",
            "At a market stall",
            "On a barbecue grill",
            "On a plate during a meal"
        ]
        
        places_scenarios=[
        "In snowy weather",
        "On a sunny day",
        "In rainy weather",
        "On a cloudy day",
        "In cloudy weather",
        "In a storm",
        "In foggy weather",
        "In a thunderstorm",
        "In hailstorm",
        "In sunrise/sunset"
        ]
        
        flowers_scenarios=["Fresh",
        "Withered",
        "Wilting",
        "Blooming",
        "Drooping",
        "Fading",
        "Blossoming",
        "Bud",
        "Yellowing",
        "Browning",
        "Shedding",
        "Partial Bloom"
        ]

        animal_scenarios = [
        "In the grass",
        "In the forest",
        "By the river",
        "On a mountaintop",
        "In the desert",
        "In the jungle",
        "On the ocean shore",
        "In a cave",
        "Amidst snow-covered peaks",
        "In a bustling city",
        "In a tranquil meadow"
        ""
        ]


        # animal_scenarios=[""]
        placeholder_token = "<"+domain+"-"+new_label[oriconcept]+", a kind of "+origin_concept[oriconcept]+">" 
        initializer_token = origin_concept[oriconcept]
        
        prompt= "a photo of "+placeholder_token
        print(prompt)
        
        # prompts=[prompt+" "+i for i in food_scenarios]
        if domain=='flower':
            prompts=[prompt+" which is "+i for i in flowers_scenarios]
        elif domain=='food':
            prompts=[prompt+" "+i for i in food_scenarios]
        elif domain=='animal':
            prompts=[prompt+" "+i for i in animal_scenarios]


        num_samples = 10 #@param {type:"number"}
        num_rows = 1 #@param {type:"number"}

        all_images = []
        images=[] 
        for i in range(std_cnt/30+1):
            for _ in range(num_rows):
                idx=np.random.randint(0,len(prompts))
                images += pipe([prompts[idx]] * num_samples, num_inference_steps=30, guidance_scale=7.5).images
                all_images.extend(images)
        cnt=0
        for image in images:
            if cnt>std_cnt/3:
                break
            image.save(save_path+"/result"+str(cnt)+".jpg")
            cnt+=1
        #grid = image_grid(all_images, num_rows, num_samples)
