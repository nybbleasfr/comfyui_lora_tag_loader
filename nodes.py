from pathlib import Path
from nodes import LoraLoader, LoraLoaderModelOnly
import folder_paths
import re

# Import ComfyUI files
import comfy.sd
import comfy.utils


class BaseLoraTagLoader:

    def __init__(self):
        self.tag_pattern: re.Pattern = re.compile("\<lora:[^\>]+\>")

    def enumerate_loras(self, model, clip, text, normalize_weight, high_to_low=False):
        # print(f"\nLoraTagLoader input text: { text }")

        founds = self.tag_pattern.findall(text)
        # print(f"\nfound lora tags: { founds }")

        if len(founds) < 1:
            return (model, clip, text)

        model_lora = model
        clip_lora = clip

        loras = []
        wModels = []
        wClips = []

        max_clip = 0.0
        max_weight = 0.0

        lora_files = folder_paths.get_filename_list("loras")
        for f in founds:
            tag = f[1:-1]
            pak: list[str] = tag.split(":")
            type = pak.pop(0)
            if type != "lora":
                continue
            name = None
            if len(pak) > 0 and len(pak[0]) > 0:
                name = pak.pop(0)
            else:
                continue
            wModel: float = 0.0
            wClip: float = 0.0
            try:
                if len(pak) > 0 and len(pak[0]) > 0:
                    wModel = float(pak.pop(0))
                if clip is not None:
                    if len(pak) > 0 and len(pak[0]) > 0:
                        wClip = float(pak.pop(0))
                    else:
                        wClip = wModel
            except ValueError:
                continue
            if name == None:
                continue

            if high_to_low:
                name = (
                    str(name)
                    .replace("high", "low")
                    .replace("HIGH", "LOW")
                    .replace("High", "Low")
                )

            print(f"Lora: {name}")
            lora_name = None
            for lora_file in lora_files:
                if Path(lora_file).name.startswith(name) or lora_file.startswith(name):
                    lora_name = lora_file
                    break

            if lora_name == None:
                print(
                    f"bypassed lora tag: { (type, name, wModel, wClip) } >> { lora_name }"
                )
                continue

            if wClip != 0.0 or wModel != 0.0:
                max_clip += abs(wClip)
                max_weight += abs(wModel)

                loras.append(lora_name)
                wModels.append(wModel)
                wClips.append(wClip)

        for idx, l in enumerate(loras):
            if normalize_weight > 0:
                weight_scale = normalize_weight / max_weight if max_weight > 0 else 1.0
                clip_scale = normalize_weight / max_clip if max_clip > 0 else 1.0
            else:
                weight_scale = 1.0
                clip_scale = 1.0
            final_weight = wModels[idx] * weight_scale
            final_clip = wClips[idx] * clip_scale
            print(
                f"Applying LORA tag: { l } weight={ round(final_weight, 3) } clip={ round(final_clip, 3) }"
            )
            yield (model_lora, clip_lora, l, final_weight, final_clip)


class LoraTagLoaderModelOnly(BaseLoraTagLoader):

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "normalize_weight": (
                    "FLOAT",
                    {"default": 0, "min": 0, "max": 100.0, "step": 0.01, "round": 0.01},
                ),
                "high_to_low": (
                    "BOOLEAN",
                    {"default": False, "label": "High to Low"},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "STRING")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"

    def load_lora(self, model, text, normalize_weight, high_to_low=False):
        for (
            model_lora,
            _,
            l,
            final_weight,
            _,
        ) in self.enumerate_loras(model, None, text, normalize_weight, high_to_low):
            model_lora = LoraLoaderModelOnly().load_lora_model_only(
                model_lora, l, final_weight
            )[0]

        plain_prompt = self.tag_pattern.sub("", text)
        return (model_lora, plain_prompt)


class LoraTagLoader(BaseLoraTagLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "normalize_weight": (
                    "FLOAT",
                    {"default": 0, "min": 0, "max": 100.0, "step": 0.01, "round": 0.01},
                ),
                "high_to_low": (
                    "BOOLEAN",
                    {"default": False, "label": "High to Low"},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "STRING")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"

    def load_lora(self, model, clip, text, normalize_weight, high_to_low):
        for (
            model_lora,
            clip_lora,
            l,
            final_weight,
            final_clip,
        ) in self.enumerate_loras(model, clip, text, normalize_weight, high_to_low):
            model_lora, clip_lora = LoraLoader().load_lora(
                model_lora, clip_lora, l, final_weight, final_clip
            )

        plain_prompt = self.tag_pattern.sub("", text)
        return (model_lora, clip_lora, plain_prompt)


NODE_CLASS_MAPPINGS = {
    "LoraTagLoader": LoraTagLoader,
    "LoraTagLoaderModelOnly": LoraTagLoaderModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Loaders
    "LoraTagLoader": "Load LoRA Tag",
    "LoraTagLoaderModelOnly": "Load LoRA Tag Model Only",
}
