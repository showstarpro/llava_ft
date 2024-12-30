import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPTextModel

# import sys
# import os
# current_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(current_path+'/control_encoder.py')

import importlib
# # from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
# def _import_file(module_name, file_path, make_importable=False):
#     spec = importlib.util.spec_from_file_location(module_name, file_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     if make_importable:
#         sys.modules[module_name] = module
#     return module

from .control_encoder import DiTVisionModel, conv_nd, zero_module
import math
from .clip_qavit import InstructCLIPVisionModel

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_contr, projector_contr, zero_model, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.vision_tower_contr_name = vision_tower_contr ## add control prompt 
        self.projector_contr_name = projector_contr
        self.zero_model_name = zero_model


     
        ###############################
        
        ### add for qa-vit
        self.instruction_embedder = None
        self.instruction_dim = 768
        self.integration_point= 'late'
        
        config = CLIPVisionConfig.from_pretrained(self.vision_tower_contr_name)
        self.vision_tower = InstructCLIPVisionModel(config=config, instruction_dim=self.instruction_dim,
                                                    integration_point=self.integration_point)

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        # here load the vision encoder, can change this 
        self.image_processor = CLIPImageProcessor.from_pretrained('/lpai/volumes/so-volume-ga/models/clip-vit-large-patch14-336')


        # pretrained_dict = CLIPVisionModel.from_pretrained(self.vision_tower_name).state_dict()
        from safetensors.torch import load_file
        pretrained_dict = load_file('/lpai/volumes/so-volume-ga/lhp/vicuna-7b-v1.5-pretrain/clip_vitl_336_control_qa_vit/vision_tower/model.safetensors')
        missing, unexpected = self.vision_tower.load_state_dict(pretrained_dict, strict=False)
        self.vision_tower = self.vision_tower.to(self.device)
        assert len(unexpected) == 0     # asserts that loading weights was as expected
        self.vision_tower.init_qavit_comps()
        if self.vision_tower is not None:
            for name, param in self.vision_tower.named_parameters():
                if 'instruct' not in name:  # qa-vit components are named with instruct and are trainables
                    param.requires_grad = False

        ### add for qa-vit #########
        
        text_tower = CLIPTextModel.from_pretrained(self.vision_tower_name)
        self.instruction_embedder = text_tower.text_model.embeddings.token_embedding.to(self.device)
        self.instruction_dim = self.instruction_embedder.weight.shape[-1]

        ########################################

        self.is_loaded = True

    def forward_features(self, images, prompts=None):
        
        image_forward_outs = self.vision_tower(pixel_values=images, output_hidden_states=True, instruct_states=prompts)

        return image_forward_outs

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            with torch.no_grad():
                image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def forward(self, images, prompt=None):
        ### add prompt for control
        with torch.no_grad():
                prompt = prompt.to(device=self.device)
                prompt_features = self.instruction_embedder(prompt)                

        if type(images) is list:
            image_features = []
            for image in images:
                # image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                # image_feature = self.feature_select(image_forward_out).to(image.dtype)

                image_forward_outs = self.forward_features(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), prompt_features)
                image_feature = self.feature_select(image_forward_outs).to(images.dtype)

                image_features.append(image_feature)
        else:
            # image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            # image_features = self.feature_select(image_forward_outs).to(images.dtype)
        
            image_forward_outs = self.forward_features(images.to(device=self.device, dtype=self.dtype), prompt_features)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)


        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
