from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .constants import TOKEN_LENGTH, DOWNLOAD_ROOT
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

from data_utils.imagenet_prompts import imagenet_templates

import copy

_tokenizer = _Tokenizer()

from data_utils.cls_to_names import *


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding[:TOKEN_LENGTH]
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x



# Text Matrix Adapter
class TextMatrixAdapter(nn.Module):
    def __init__(
        self,

        args,

        device,
        classnames,
        batch_size,
        arch="ViT-L/14",

        n_ctx=16, 
        ctx_init=None, 
        ctx_position='end', 
        learned_cls=False
    ):
        super().__init__()

        # ------------------------
        # Load CLIP
        # ------------------------
        self.clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)

        self.device = device
        self.classnames = classnames

        self.image_encoder = self.clip.visual

        self.embed_dim = self.image_encoder.output_dim

        # logit scale (frozen)
        self.logit_scale = self.clip.logit_scale.detach()

        # ------------------------
        # Text prototypes (frozen)
        # ------------------------
        with torch.no_grad():
            if args.coop_init:
                assert args.coop_ckpt is not None, "Please provide --coop_ckpt"

                coop_ctx = self.load_coop_ctx(args.coop_ckpt)

                self.text_prototypes = self.build_text_prototypes_from_coop(coop_ctx)
            elif args.ensemble:
                self.text_prototypes = self.build_text_prototypes(imagenet_templates)
            else:
                self.text_prototypes = self.build_text_prototypes(ctx_init = ctx_init)

        self.W = nn.Parameter(
            torch.eye(self.embed_dim, device=device, dtype=self.dtype)
        )

        self.register_buffer(
            "W_init", self.W.detach().clone()
        )

        # Freeze everything except Weight matrix
        for name, param in self.named_parameters():
            if "W" not in name:
                param.requires_grad_(False)

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype
    
    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def _l2_normalize(self, x, eps=1e-8):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def reset(self):
        """Episodic reset"""
        with torch.no_grad():
            self.W.copy_(self.W_init)
    
    # --------------------------------------------------
    # Load CoOp context vectors
    # --------------------------------------------------
    def load_coop_ctx(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # CoOp stores ctx here
        coop_ctx = ckpt["state_dict"]["ctx"]   # (n_ctx, dim)

        return coop_ctx.to(self.device)

    # --------------------------------------------------
    # Text prototype construction
    # --------------------------------------------------
    def build_text_prototypes_from_coop(self, coop_ctx):
        """
        Build text prototypes using CoOp soft prompts.

        coop_ctx: (n_ctx, dim)
        returns: (num_classes, dim)
        """

        classnames = [c.replace("_", " ") for c in self.classnames]
        device = self.device
        clip = self.clip

        prompts = []
        eot_indices = []

        n_ctx = coop_ctx.shape[0]
        embed_dim = coop_ctx.shape[1]
        TOKEN_LENGTH = 77

        for name in classnames:
            text = name + "."
            tokens = tokenize(text).to(device) # (1, 77)
            token_ids = tokens[0]

            with torch.no_grad():
                emb = clip.token_embedding(tokens).type(self.dtype)  # (1, 77, dim)

            # -------- split original tokens --------
            prefix = emb[:, :1, :]  # SOS

            eot_pos = token_ids.argmax().item() # original EOT position
            class_and_eos = emb[:, 1:eot_pos+1, :] # class tokens + EOS

            # -------- construct new prompt --------
            prompt = torch.cat(
                [prefix, coop_ctx.unsqueeze(0), class_and_eos],
                dim=1
            )

            # -------- pad to 77 tokens --------
            pad_len = TOKEN_LENGTH - prompt.shape[1]
            if pad_len > 0:
                pad = torch.zeros(
                    1, pad_len, embed_dim,
                    device=device, dtype=prompt.dtype
                )
                prompt = torch.cat([prompt, pad], dim=1)

            prompts.append(prompt)

            # -------- compute new EOT position --------
            new_eot = 1 + n_ctx + (eot_pos - 1)
            eot_indices.append(new_eot)

        prompts = torch.cat(prompts, dim=0)                     # (C, 77, dim)
        eot_indices = torch.tensor(eot_indices, device=device)  # (C,)

        # -------- CLIP text transformer --------
        text_encoder = TextEncoder(clip).to(device)

        with torch.no_grad():
            x = prompts + text_encoder.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)
            x = text_encoder.transformer(x)
            x = x.permute(1, 0, 2)
            x = text_encoder.ln_final(x).type(self.dtype)

            text_features = x[torch.arange(x.shape[0]), eot_indices] @ text_encoder.text_projection

        # -------- normalize --------
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def build_text_prototypes(self, templates=None, ctx_init="a_photo_of_a"):
        if templates is None:
            templates = [ctx_init.replace("_", " ").strip() + " {}."]

        return self._build_from_templates(templates)


    def _build_from_templates(self, templates):
        """
        Build text prototypes using multiple prompt templates (prompt ensemble).

        templates: list of strings, e.g. imagenet_templates
                each containing exactly one {} placeholder.
        """

        all_text_features = []

        for template in templates:
            texts = [template.format(c.replace("_", " ")) for c in self.classnames]

            tokens = tokenize(texts).to(self.device)

            with torch.no_grad():
                t_features = self.clip.encode_text(tokens)

            # Normalize per template
            t_features = t_features / t_features.norm(dim=-1, keepdim=True)

            all_text_features.append(t_features)

        # Shape: (num_templates, n_cls, dim)
        all_text_features = torch.stack(all_text_features, dim=0)

        # Average over templates
        text_prototypes = all_text_features.mean(dim=0)

        # Final normalization
        text_prototypes = text_prototypes / text_prototypes.norm(dim=-1, keepdim=True)

        return text_prototypes


    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    def forward(self, images):
        """
        images: (B, 3, H, W)
        """

        # -------- Image side --------
        img_feat = self.image_encoder(images.type(self.dtype)) # (B x embed_dim)
        img_feat = self._l2_normalize(img_feat)

        # -------- Text side --------
        txt_feat = self.text_prototypes # (Classes, embed_dim)
        txt_feat = txt_feat @ self.W.T
        txt_feat = self._l2_normalize(txt_feat)

        # -------- Similarity --------
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_feat @ txt_feat.t()

        return logits

    def forward_features(self, input):
        """
        Returns:
            image_features : (B, D)   shifted + normalized image embeddings
            text_features  : (C, D)   shifted + normalized class text prototypes
            logit_scale    : scalar   CLIP temperature
        """
        with torch.no_grad():
            # ------------------
            # Image side
            # ------------------
            image_features = self.image_encoder(input.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # ------------------
            # Text side
            # ------------------
            text_features = self.text_prototypes
            text_features = text_features @ self.W.T
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # ------------------
            # Temperature
            # ------------------
            logit_scale = self.logit_scale.exp()

        return image_features, text_features, logit_scale

def gettextmatrixadapter(args, clip_arch, classnames, device, n_ctx, ctx_init, learned_cls=False):
    model = TextMatrixAdapter(
        args,
        device=device,
        classnames=classnames,
        batch_size=None,
        arch=clip_arch,
        n_ctx=n_ctx,
        ctx_init=ctx_init,
        learned_cls=learned_cls
    )
    
    return model



# Image Matrix Adapter
class ImageMatrixAdapter(nn.Module):
    """
    CLIP with:
    """

    def __init__(
        self,

        args,

        device,
        classnames,
        batch_size,
        arch="ViT-L/14",

        n_ctx=16, 
        ctx_init=None, 
        ctx_position='end', 
        learned_cls=False,
    ):
        super().__init__()

        # ------------------------
        # Load CLIP
        # ------------------------
        self.clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)

        self.device = device
        self.classnames = classnames

        self.image_encoder = self.clip.visual

        self.embed_dim = self.image_encoder.output_dim

        # logit scale (frozen)
        self.logit_scale = self.clip.logit_scale.detach()

        # ------------------------
        # Text prototypes (frozen)
        # ------------------------
        with torch.no_grad():
            if args.coop_init:
                assert args.coop_ckpt is not None, "Please provide --coop_ckpt"

                coop_ctx = self.load_coop_ctx(args.coop_ckpt)

                self.text_prototypes = self.build_text_prototypes_from_coop(coop_ctx)

                # print("Prototype shape:", self.text_prototypes.shape)
                # print("Mean norm:", self.text_prototypes.norm(dim=1).mean().item())
            elif args.ensemble:
                self.text_prototypes = self.build_text_prototypes(imagenet_templates)
            else:
                self.text_prototypes = self.build_text_prototypes(ctx_init = ctx_init)

        self.W = nn.Parameter(
            torch.eye(self.embed_dim, device=device, dtype=self.dtype)
        )

        # Save initial states for episodic reset
        self.register_buffer(
            "W_init", self.W.detach().clone()
        )

        # Freeze everything except Weight matrix
        for name, param in self.named_parameters():
            if "W" not in name:
                param.requires_grad_(False)

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype
    
    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def _l2_normalize(self, x, eps=1e-8):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def reset(self):
        """Episodic reset"""
        with torch.no_grad():
            self.W.copy_(self.W_init)


    # --------------------------------------------------
    # Load CoOp context vectors
    # --------------------------------------------------
    def load_coop_ctx(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # CoOp stores ctx here
        coop_ctx = ckpt["state_dict"]["ctx"]   # (n_ctx, dim)

        return coop_ctx.to(self.device)


    # --------------------------------------------------
    # Text prototype construction
    # --------------------------------------------------
    def build_text_prototypes_from_coop(self, coop_ctx):
        """
        Build text prototypes using CoOp soft prompts.

        coop_ctx: (n_ctx, dim)
        returns: (num_classes, dim)
        """

        classnames = [c.replace("_", " ") for c in self.classnames]
        device = self.device
        clip = self.clip

        prompts = []
        eot_indices = []

        n_ctx = coop_ctx.shape[0]
        embed_dim = coop_ctx.shape[1]
        TOKEN_LENGTH = 77

        for name in classnames:
            text = name + "."
            tokens = tokenize(text).to(device)  # (1, 77)
            token_ids = tokens[0]

            with torch.no_grad():
                emb = clip.token_embedding(tokens).type(self.dtype)  # (1, 77, dim)

            # -------- split original tokens --------
            prefix = emb[:, :1, :]  # SOS

            eot_pos = token_ids.argmax().item()             # original EOT position
            class_and_eos = emb[:, 1:eot_pos+1, :]          # class tokens + EOS

            # -------- construct new prompt --------
            prompt = torch.cat(
                [prefix, coop_ctx.unsqueeze(0), class_and_eos],
                dim=1
            )

            # -------- pad to 77 tokens --------
            pad_len = TOKEN_LENGTH - prompt.shape[1]
            if pad_len > 0:
                pad = torch.zeros(
                    1, pad_len, embed_dim,
                    device=device, dtype=prompt.dtype
                )
                prompt = torch.cat([prompt, pad], dim=1)

            prompts.append(prompt)

            # -------- compute new EOT position --------
            new_eot = 1 + n_ctx + (eot_pos - 1)
            eot_indices.append(new_eot)

        prompts = torch.cat(prompts, dim=0)                     # (C, 77, dim)
        eot_indices = torch.tensor(eot_indices, device=device)  # (C,)

        # -------- CLIP text transformer --------
        text_encoder = TextEncoder(clip).to(device)

        with torch.no_grad():
            x = prompts + text_encoder.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)
            x = text_encoder.transformer(x)
            x = x.permute(1, 0, 2)
            x = text_encoder.ln_final(x).type(self.dtype)

            text_features = x[torch.arange(x.shape[0]), eot_indices] @ text_encoder.text_projection

        # -------- normalize --------
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def build_text_prototypes(self, templates=None, ctx_init="a_photo_of_a"):
        if templates is None:
            templates = [ctx_init.replace("_", " ").strip() + " {}."]

        return self._build_from_templates(templates)

    def _build_from_templates(self, templates):
        """
        Build text prototypes using multiple prompt templates (prompt ensemble).

        templates: list of strings, e.g. imagenet_templates
                each containing exactly one {} placeholder.
        """

        all_text_features = []

        for template in templates:
            texts = [template.format(c.replace("_", " ")) for c in self.classnames]

            tokens = tokenize(texts).to(self.device)

            with torch.no_grad():
                t_features = self.clip.encode_text(tokens)

            # Normalize per template
            t_features = t_features / t_features.norm(dim=-1, keepdim=True)

            all_text_features.append(t_features)

        # Shape: (num_templates, n_cls, dim)
        all_text_features = torch.stack(all_text_features, dim=0)

        # Average over templates
        text_prototypes = all_text_features.mean(dim=0)

        # Final normalization
        text_prototypes = text_prototypes / text_prototypes.norm(dim=-1, keepdim=True)

        return text_prototypes


    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    def forward(self, images):
        """
        images: (B, 3, H, W)
        """

        # -------- Image side --------
        img_feat = self.image_encoder(images.type(self.dtype)) # (B x embed_dim)
        img_feat = img_feat @ self.W.T # (B x embed_dim)
        img_feat = self._l2_normalize(img_feat)

        # -------- Text side --------
        txt_feat = self.text_prototypes
        txt_feat = self._l2_normalize(txt_feat)

        # -------- Similarity --------
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_feat @ txt_feat.t()

        return logits

    def forward_features(self, input):
        """
        Returns:
            image_features : (B, D)   shifted + normalized image embeddings
            text_features  : (C, D)   shifted + normalized class text prototypes
            logit_scale    : scalar   CLIP temperature
        """
        with torch.no_grad():
            # ------------------
            # Image side
            # ------------------
            image_features = self.image_encoder(input.type(self.dtype))
            image_features = image_features @ self.W.T
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # ------------------
            # Text side
            # ------------------
            text_features = self.text_prototypes
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # ------------------
            # Temperature
            # ------------------
            logit_scale = self.logit_scale.exp()

        return image_features, text_features, logit_scale

def getimagematrixadapter(args, clip_arch, classnames, device, n_ctx, ctx_init, learned_cls=False):
    model = ImageMatrixAdapter(
        args,
        device=device,
        classnames=classnames,
        batch_size=None,
        arch=clip_arch,
        n_ctx=n_ctx,
        ctx_init=ctx_init,
        learned_cls=learned_cls
    )
    
    return model



# ZERO-SHOT CLIP
class ZeroShotCLIP(nn.Module):
    """
    CLIP with:
    """

    def __init__(
        self,

        args,

        device,
        classnames,
        batch_size,
        arch="ViT-L/14",

        n_ctx=16, 
        ctx_init=None, 
        ctx_position='end', 
        learned_cls=False,
    ):
        super().__init__()

        # ------------------------
        # Load CLIP
        # ------------------------
        self.clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)

        self.device = device
        self.classnames = classnames

        self.image_encoder = self.clip.visual

        self.embed_dim = self.image_encoder.output_dim

        # logit scale (frozen)
        self.logit_scale = self.clip.logit_scale.detach()

        # ------------------------
        # Text prototypes (frozen)
        # ------------------------
        with torch.no_grad():
            if args.coop_init:
                assert args.coop_ckpt is not None, "Please provide --coop_ckpt"

                coop_ctx = self.load_coop_ctx(args.coop_ckpt)

                self.text_prototypes = self.build_text_prototypes_from_coop(coop_ctx)

                # print("Prototype shape:", self.text_prototypes.shape)
                # print("Mean norm:", self.text_prototypes.norm(dim=1).mean().item())
            elif args.ensemble:
                self.text_prototypes = self.build_text_prototypes(imagenet_templates)
            else:
                self.text_prototypes = self.build_text_prototypes(ctx_init = ctx_init)

        # Freeze everything
        for name, param in self.named_parameters():
            param.requires_grad_(False)

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype
    
    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def _l2_normalize(self, x, eps=1e-8):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def reset(self):
        """Episodic reset"""
        with torch.no_grad():
            self.W.copy_(self.W_init)


    # --------------------------------------------------
    # Load CoOp context vectors
    # --------------------------------------------------
    def load_coop_ctx(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # CoOp stores ctx here
        coop_ctx = ckpt["state_dict"]["ctx"]   # (n_ctx, dim)

        return coop_ctx.to(self.device)


    # --------------------------------------------------
    # Text prototype construction
    # --------------------------------------------------
    def build_text_prototypes_from_coop(self, coop_ctx):
        """
        Build text prototypes using CoOp soft prompts.

        coop_ctx: (n_ctx, dim)
        returns: (num_classes, dim)
        """

        classnames = [c.replace("_", " ") for c in self.classnames]
        device = self.device
        clip = self.clip

        prompts = []
        eot_indices = []

        n_ctx = coop_ctx.shape[0]
        embed_dim = coop_ctx.shape[1]
        TOKEN_LENGTH = 77

        for name in classnames:
            text = name + "."
            tokens = tokenize(text).to(device)  # (1, 77)
            token_ids = tokens[0]

            with torch.no_grad():
                emb = clip.token_embedding(tokens).type(self.dtype)  # (1, 77, dim)

            # -------- split original tokens --------
            prefix = emb[:, :1, :]  # SOS

            eot_pos = token_ids.argmax().item()             # original EOT position
            class_and_eos = emb[:, 1:eot_pos+1, :]          # class tokens + EOS

            # -------- construct new prompt --------
            prompt = torch.cat(
                [prefix, coop_ctx.unsqueeze(0), class_and_eos],
                dim=1
            )

            # -------- pad to 77 tokens --------
            pad_len = TOKEN_LENGTH - prompt.shape[1]
            if pad_len > 0:
                pad = torch.zeros(
                    1, pad_len, embed_dim,
                    device=device, dtype=prompt.dtype
                )
                prompt = torch.cat([prompt, pad], dim=1)

            prompts.append(prompt)

            # -------- compute new EOT position --------
            new_eot = 1 + n_ctx + (eot_pos - 1)
            eot_indices.append(new_eot)

        prompts = torch.cat(prompts, dim=0)                     # (C, 77, dim)
        eot_indices = torch.tensor(eot_indices, device=device)  # (C,)

        # -------- CLIP text transformer --------
        text_encoder = TextEncoder(clip).to(device)

        with torch.no_grad():
            x = prompts + text_encoder.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)
            x = text_encoder.transformer(x)
            x = x.permute(1, 0, 2)
            x = text_encoder.ln_final(x).type(self.dtype)

            text_features = x[torch.arange(x.shape[0]), eot_indices] @ text_encoder.text_projection

        # -------- normalize --------
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def build_text_prototypes(self, templates=None, ctx_init="a_photo_of_a"):
        if templates is None:
            templates = [ctx_init.replace("_", " ").strip() + " {}."]

        return self._build_from_templates(templates)

    def _build_from_templates(self, templates):
        """
        Build text prototypes using multiple prompt templates (prompt ensemble).

        templates: list of strings, e.g. imagenet_templates
                each containing exactly one {} placeholder.
        """

        all_text_features = []

        for template in templates:
            texts = [template.format(c.replace("_", " ")) for c in self.classnames]

            tokens = tokenize(texts).to(self.device)

            with torch.no_grad():
                t_features = self.clip.encode_text(tokens)

            # Normalize per template
            t_features = t_features / t_features.norm(dim=-1, keepdim=True)

            all_text_features.append(t_features)

        # Shape: (num_templates, n_cls, dim)
        all_text_features = torch.stack(all_text_features, dim=0)

        # Average over templates
        text_prototypes = all_text_features.mean(dim=0)

        # Final normalization
        text_prototypes = text_prototypes / text_prototypes.norm(dim=-1, keepdim=True)

        return text_prototypes


    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    def forward(self, images):
        """
        images: (B, 3, H, W)
        """

        # -------- Image side --------
        img_feat = self.image_encoder(images.type(self.dtype)) # (B x embed_dim)
        img_feat = self._l2_normalize(img_feat)

        # -------- Text side --------
        txt_feat = self.text_prototypes
        txt_feat = self._l2_normalize(txt_feat)

        # -------- Similarity --------
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_feat @ txt_feat.t()

        return logits

def getzeroshotclip(args, clip_arch, classnames, device, n_ctx, ctx_init, learned_cls=False):
    model = ZeroShotCLIP(
        args,
        device=device,
        classnames=classnames,
        batch_size=None,
        arch=clip_arch,
        n_ctx=n_ctx,
        ctx_init=ctx_init,
        learned_cls=learned_cls
    )
    
    return model