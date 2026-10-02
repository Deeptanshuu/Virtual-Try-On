"""Headless IDM-VTON try-on engine.

Loads the models once and exposes the try-on as separate steps, so the Gradio
app, the API and batch experiment scripts all share one code path:

    engine = TryOnEngine()
    person = engine.prepare_person(human_img)          # pose, parsing, mask, DensePose
    prompts = engine.encode_prompts("blue t-shirt")
    for seed in seeds:                                 # preprocessing is reused per sample
        image = engine.generate(person, garment_img, prompts, seed=seed)
        result = engine.paste_back(person, image)

Run one process per GPU (CUDA_VISIBLE_DEVICES=0 / =1) to use several GPUs.
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

import apply_net
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from utils_mask import get_mask_location

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = 'yisol/IDM-VTON'

# Standard size on which model is trained
WIDTH = 768
HEIGHT = 1024
POSE_WIDTH = WIDTH // 2
POSE_HEIGHT = HEIGHT // 2

DEFAULT_PROMPT = "model is wearing a garment"
DEFAULT_GARMENT_PROMPT = "a photo of a garment"
NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality"

DENSEPOSE_CONFIG = os.path.join(REPO_ROOT, 'configs', 'densepose_rcnn_R_50_FPN_s1x.yaml')
DENSEPOSE_CKPT = os.path.join(REPO_ROOT, 'ckpt', 'densepose', 'model_final_162be9.pkl')
PREPROCESS_CKPTS = [
    DENSEPOSE_CKPT,
    os.path.join(REPO_ROOT, 'ckpt', 'humanparsing', 'parsing_atr.onnx'),
    os.path.join(REPO_ROOT, 'ckpt', 'humanparsing', 'parsing_lip.onnx'),
    os.path.join(REPO_ROOT, 'ckpt', 'openpose', 'ckpts', 'body_pose_model.pth'),
]


def check_checkpoints():
    """Fail early if a checkpoint in ckpt/ is a Git LFS pointer instead of the real file.

    The copies committed to this repo are pointers; loading them gives cryptic onnx/pickle errors.
    """
    stubs = [
        os.path.relpath(path, REPO_ROOT)
        for path in PREPROCESS_CKPTS
        if os.path.exists(path) and os.path.getsize(path) < 1024
    ]
    if stubs:
        raise RuntimeError(
            "These checkpoints are Git LFS pointer files, not model weights: " + ", ".join(stubs)
            + ". Download the real files from https://huggingface.co/spaces/yisol/IDM-VTON/tree/main/ckpt"
        )


def _no_progress(*args, **kwargs):
    pass


def is_cropping_required(width, height):
    # If aspect ratio is 1.33, which is same as standard 3x4 ( 768x1024 ), then no need to crop, else crop
    aspect_ratio = round(height / width, 2)
    return aspect_ratio != 1.33


@dataclass
class PreparedPerson:
    """Everything derived from the person image; independent of garment and seed."""
    original: Image.Image          # input image, RGB, original size
    image: Image.Image             # person image at model size (WIDTH x HEIGHT)
    mask: Image.Image              # inpainting mask at model size
    mask_gray: Image.Image         # person image with the masked area greyed out
    pose_img: Image.Image          # DensePose render at model size
    crop_offset: Optional[Tuple[int, int]] = None  # where the 3:4 crop sits in `original`
    crop_size: Optional[Tuple[int, int]] = None


@dataclass
class PromptEmbeds:
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    negative_pooled_prompt_embeds: torch.Tensor
    garment_embeds: torch.Tensor


class TryOnEngine:
    def __init__(self, base_model=BASE_MODEL, device=None):
        check_checkpoints()
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float16

        unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=dtype)
        tokenizer_one = AutoTokenizer.from_pretrained(
            base_model, subfolder="tokenizer", revision=None, use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            base_model, subfolder="tokenizer_2", revision=None, use_fast=False,
        )
        noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
        text_encoder_one = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=dtype)
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            base_model, subfolder="text_encoder_2", torch_dtype=dtype,
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model, subfolder="image_encoder", torch_dtype=dtype,
        )
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype)
        unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            base_model, subfolder="unet_encoder", torch_dtype=dtype,
        )

        for model in (unet_encoder, image_encoder, vae, unet, text_encoder_one, text_encoder_two):
            model.requires_grad_(False)

        self.pipe = TryonPipeline.from_pretrained(
            base_model,
            unet=unet,
            vae=vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            scheduler=noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=dtype,
        )
        self.pipe.unet_encoder = unet_encoder

        self.parsing_model = Parsing(0)
        self.openpose_model = OpenPose(0)
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self._on_device = False

    def to_device(self):
        if self._on_device:
            return
        self.openpose_model.preprocessor.body_estimation.model.to(self.device)
        self.pipe.to(self.device)
        self.pipe.unet_encoder.to(self.device)
        self._on_device = True

    def prepare_person(self, human_img, garment_category="upper_body", arm_width="dc", progress=_no_progress):
        """Crop/resize the person image and compute its mask and DensePose map."""
        self.to_device()
        original = human_img.convert("RGB")
        width, height = original.size

        crop_offset = crop_size = None
        if is_cropping_required(width, height):
            # Crop to a 3x4 aspect ratio; paste_back() puts the result back into the original.
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            right = (width + target_width) / 2
            # for Landmark, model sizes are 594x879, so we need to reduce the height. In some case the garment on the model is
            # also getting removed when reducing size from bottom. So we will only reduce height from top for now
            top = height - target_height
            bottom = height
            cropped = original.crop((left, top, right, bottom))
            crop_offset = (int(left), int(top))
            crop_size = cropped.size
            image = cropped.resize((WIDTH, HEIGHT))
        else:
            image = original.resize((WIDTH, HEIGHT))

        pose_input = image.resize((POSE_WIDTH, POSE_HEIGHT))
        progress(0.2, desc="Analyzing pose and body parts...")
        keypoints = self.openpose_model(pose_input)
        progress(0.25, desc="Parsing human body segments...")
        model_parse, _ = self.parsing_model(pose_input)
        progress(0.3, desc="Generating clothing mask...")
        # get_mask_location works at 384x512, so scale the mask up to model size
        mask, _ = get_mask_location(arm_width, garment_category, model_parse, keypoints)
        mask = mask.resize((WIDTH, HEIGHT))

        mask_gray = (1 - transforms.ToTensor()(mask)) * self.tensor_transform(image)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

        progress(0.35, desc="Processing dense pose estimation...")
        pose_img = self._densepose(pose_input)

        return PreparedPerson(
            original=original,
            image=image,
            mask=mask,
            mask_gray=mask_gray,
            pose_img=pose_img,
            crop_offset=crop_offset,
            crop_size=crop_size,
        )

    def _densepose(self, pose_input):
        image = _apply_exif_orientation(pose_input)
        image = convert_PIL_to_numpy(image, format="BGR")
        args = apply_net.create_argument_parser().parse_args((
            'show', DENSEPOSE_CONFIG, DENSEPOSE_CKPT, 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', self.device,
        ))
        pose_img = args.func(args, image)
        pose_img = pose_img[:, :, ::-1]
        return Image.fromarray(pose_img).resize((WIDTH, HEIGHT))

    def encode_prompts(self, prompt_text=""):
        """Encode the try-on prompt and the garment prompt; reuse the result across samples."""
        self.to_device()
        description = (prompt_text or "").strip()
        prompt = description or DEFAULT_PROMPT
        garment_prompt = f"{DEFAULT_GARMENT_PROMPT}, {description}" if description else DEFAULT_GARMENT_PROMPT

        with torch.no_grad(), torch.cuda.amp.autocast(), torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=NEGATIVE_PROMPT,
            )
            garment_embeds, _, _, _ = self.pipe.encode_prompt(
                [garment_prompt],
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=[NEGATIVE_PROMPT],
            )

        return PromptEmbeds(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            garment_embeds=garment_embeds,
        )

    def generate(
        self,
        person: PreparedPerson,
        garment_img: Image.Image,
        prompts: PromptEmbeds,
        seed=-1,
        steps=30,
        guidance_scale=2.0,
        step_callback: Optional[Callable] = None,
    ):
        """Run one diffusion sample. Returns the try-on image at model size.

        seed=-1 or None samples without a fixed generator (not reproducible).
        step_callback(step, timestep, latents) is called after every denoising step.
        """
        self.to_device()
        device, dtype = self.device, torch.float16
        garment = garment_img.convert("RGB").resize((WIDTH, HEIGHT))
        pose_tensor = self.tensor_transform(person.pose_img).unsqueeze(0).to(device, dtype)
        garment_tensor = self.tensor_transform(garment).unsqueeze(0).to(device, dtype)

        if seed is None or seed == -1:
            generator = None
        else:
            generator = torch.Generator(device).manual_seed(int(seed))

        with torch.no_grad(), torch.cuda.amp.autocast(), torch.inference_mode():
            images = self.pipe(
                prompt_embeds=prompts.prompt_embeds.to(device, dtype),
                negative_prompt_embeds=prompts.negative_prompt_embeds.to(device, dtype),
                pooled_prompt_embeds=prompts.pooled_prompt_embeds.to(device, dtype),
                negative_pooled_prompt_embeds=prompts.negative_pooled_prompt_embeds.to(device, dtype),
                num_inference_steps=steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=prompts.garment_embeds.to(device, dtype),
                cloth=garment_tensor,
                mask_image=person.mask,
                image=person.image,
                height=HEIGHT,
                width=WIDTH,
                ip_adapter_image=garment,
                guidance_scale=guidance_scale,
                callback=step_callback,
                callback_steps=1,
            )[0]
        return images[0]

    def paste_back(self, person: PreparedPerson, image: Image.Image):
        """Undo the 3:4 crop: paste the model-size result back into a copy of the original image."""
        if person.crop_offset is None:
            return image
        result = person.original.copy()
        result.paste(image.resize(person.crop_size), person.crop_offset)
        return result

    def tryon(
        self,
        human_img,
        garment_img,
        prompt_text="",
        steps=30,
        seed=-1,
        garment_category="upper_body",
        arm_width="dc",
        progress=_no_progress,
    ):
        """Full single-sample try-on. Returns (result image, masked person image)."""
        progress(0.05, desc="Loading models to device...")
        self.to_device()
        progress(0.1, desc="Processing input images...")
        person = self.prepare_person(human_img, garment_category, arm_width, progress=progress)
        progress(0.4, desc="Encoding text prompts...")
        prompts = self.encode_prompts(prompt_text)

        def diffusion_progress_callback(step, timestep, latents):
            # Progress from 55% to 90% during diffusion
            progress(0.55 + (step / steps) * 0.35, desc=f"Diffusion step {step + 1}/{steps}")

        progress(0.55, desc=f"Starting diffusion process ({steps} steps)...")
        image = self.generate(
            person, garment_img, prompts, seed=seed, steps=steps, step_callback=diffusion_progress_callback,
        )
        progress(0.9, desc="Post-processing result...")
        return self.paste_back(person, image), person.mask_gray
