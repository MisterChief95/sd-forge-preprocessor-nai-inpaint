import numpy as np

from modules_forge.shared import add_supported_preprocessor
from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter


# Original implementation from:
# https://github.com/Panchovix/stable-diffusion-webui-reForge
class PreprocessorInpaintNoobAIXL(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'inpaint_noobai_xl'
        self.tags = ['Inpaint']
        self.model_filename_filters = ['inpaint', 'noobai']
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.fill_mask_with_one_when_resize_and_fill = True
        self.expand_mask_when_resize_and_fill = True

    def __call__(self, input_image, resolution=512, slider_1=None, slider_2=None, slider_3=None, input_mask=None, **kwargs):
        if input_mask is None:
            return input_image

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image)
        if not isinstance(input_mask, np.ndarray):
            input_mask = np.array(input_mask)

        mask = input_mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        # Create a copy of the input image
        result = input_image.copy()

        # Convert mask to proper shape if needed
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)
        if mask.shape[-1] == 1:
            mask = np.repeat(mask, 3, axis=-1)

        mask_indices = mask > 0.5
        result[mask_indices] = 0.0

        return result

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        mask = mask.round()
        mixed_cond = cond.clone()
        mixed_cond = mixed_cond * (1.0 - mask)

        return mixed_cond, None
    

add_supported_preprocessor(PreprocessorInpaintNoobAIXL())
