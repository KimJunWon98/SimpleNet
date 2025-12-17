import logging
import os
import pickle
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

import common
import metrics
from utils import plot_segmentation_images


try:
    import pseudo_anomaly
except Exception as e:
    pseudo_anomaly = None
    _PSEUDO_IMPORT_ERR = e

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


def _denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    """x: normalized (ImageNet mean/std) -> return [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    y = x * std + mean
    return y.clamp(0.0, 1.0)


def _norm_from_01(x01: torch.Tensor) -> torch.Tensor:
    """x01: [0,1] -> normalized (ImageNet mean/std)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


def _save_tensor_img_01(img01: torch.Tensor, out_path: str) -> None:
    """img01: (3,H,W) float in [0,1]"""
    from PIL import Image

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = (img01.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # HWC
    Image.fromarray(arr).save(out_path)


def _save_tensor_mask_01(mask01: torch.Tensor, out_path: str) -> None:
    """mask01: (1,H,W) or (H,W) float in [0,1]"""
    from PIL import Image

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    m = mask01.detach().cpu()
    if m.ndim == 3:
        m = m[0]
    arr = (m.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(out_path)


class Discriminator(torch.nn.Module):
    """Patch-level discriminator: (N,D)->(N,1). Used for noise-disc, pseudo-cls, pseudo-seg."""
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()
        _hidden = in_planes if hidden is None else hidden

        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module(
                f"block{i+1}",
                torch.nn.Sequential(
                    torch.nn.Linear(_in, _hidden),
                    torch.nn.BatchNorm1d(_hidden),
                    torch.nn.LeakyReLU(0.2),
                ),
            )

        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        if out_planes is None:
            out_planes = in_planes

        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc", torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu", torch.nn.LeakyReLU(0.2))

        self.apply(init_weight)

    def forward(self, x):
        return self.layers(x)


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class SimpleNet(torch.nn.Module):
    def __init__(self, device):
        super(SimpleNet, self).__init__()
        self.device = device

    # ---------------------------
    # helpers
    # ---------------------------
    @staticmethod
    def _patch_logits_to_image_logit(patch_logits: torch.Tensor, batch_size: int, topk: int) -> torch.Tensor:
        """
        patch_logits: (B*P, 1) or (B*P,)
        return: (B,)  (topk=1 => max, topk>1 => mean(top-k))
        """
        if patch_logits.ndim == 2 and patch_logits.size(-1) == 1:
            patch_logits = patch_logits.squeeze(-1)
        x = patch_logits.view(batch_size, -1)

        if topk <= 1:
            return x.max(dim=1).values
        k = min(int(topk), x.shape[1])
        return torch.topk(x, k, dim=1).values.mean(dim=1)

    @staticmethod
    def _downsample_mask_to_patch_grid(mask01: torch.Tensor, patch_hw: tuple[int, int]) -> torch.Tensor:
        """
        mask01: (B,1,H,W) in [0,1]
        return: (B,1,Hp,Wp) in [0,1]
        """
        Hp, Wp = int(patch_hw[0]), int(patch_hw[1])
        return F.interpolate(mask01, size=(Hp, Wp), mode="bilinear", align_corners=False).clamp(0, 1)

    # ---------------------------
    # generic weight scheduler
    # ---------------------------
    def _schedule_weight(
        self,
        meta_epoch: int,
        enabled: bool,
        w0: float,
        mode: str,
        wmin: float,
        decay_start: int,
        decay_end: int | None,
        off_after: int | None,
    ) -> float:
        if not enabled:
            return 0.0
        if off_after is not None and meta_epoch >= int(off_after):
            return 0.0

        w0 = float(w0)
        wmin = float(wmin)
        mode = str(mode).lower()

        if mode == "constant":
            return w0

        ds = int(decay_start)
        de = decay_end
        if de is None:
            de = max(ds, int(getattr(self, "meta_epochs", 1)) - 1)
        de = int(de)

        if meta_epoch <= ds:
            return w0
        if meta_epoch >= de:
            return wmin

        T = max(1, de - ds)
        t = float(meta_epoch - ds) / float(T)

        if mode == "linear":
            return w0 + (wmin - w0) * t
        if mode == "cosine":
            return wmin + 0.5 * (w0 - wmin) * (1.0 + math.cos(math.pi * t))
        if mode == "off_after":
            # handled by off_after
            return w0

        return w0

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        embedding_size=None,
        meta_epochs=1,
        aed_meta_epochs=1,
        gan_epochs=1,
        noise_std=0.05,
        mix_noise=1,
        noise_type="GAU",
        dsc_layers=2,
        dsc_hidden=None,
        dsc_margin=0.8,
        dsc_lr=0.0002,
        train_backbone=True,
        auto_noise=0,
        cos_lr=False,
        lr=1e-3,
        pre_proj=0,
        proj_layer_type=0,

        # -------------------------
        # discriminator toggles
        # -------------------------
        use_noise_discriminator=True,   # original SimpleNet noise discriminator
        use_pseudo_cls_head=True,       # pseudo classification discriminator
        use_pseudo_seg_head=False,      # pseudo segmentation discriminator (needs pseudo masks)

        # -------------------------
        # loss weights + schedules
        # -------------------------
        # noise disc weight schedule
        noise_weight=1.0,
        noise_weight_mode="constant",   # constant|linear|cosine|off_after
        noise_weight_min=1.0,
        noise_weight_decay_start=0,
        noise_weight_decay_end=None,
        noise_weight_off_after=None,

        # pseudo cls weight schedule (기존 pseudo_weight 역할)
        pseudo_cls_weight=0.1,
        pseudo_cls_weight_mode="linear",
        pseudo_cls_weight_min=0.01,
        pseudo_cls_weight_decay_start=0,
        pseudo_cls_weight_decay_end=15,
        pseudo_cls_weight_off_after=None,

        # pseudo seg weight schedule
        pseudo_seg_weight=0.1,
        pseudo_seg_weight_mode="linear",
        pseudo_seg_weight_min=0.01,
        pseudo_seg_weight_decay_start=0,
        pseudo_seg_weight_decay_end=15,
        pseudo_seg_weight_off_after=None,

        # -------------------------
        # pseudo image logging
        # -------------------------
        pseudo_save_images=False,
        pseudo_save_every=3,
        pseudo_save_max_per_epoch=30,

        # pseudo discriminator shapes
        pseudo_cls_dsc_layers=2,
        pseudo_cls_dsc_hidden=None,
        pseudo_cls_logit_topk=1,

        pseudo_seg_dsc_layers=2,
        pseudo_seg_dsc_hidden=None,

        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.embedding_size = (
            embedding_size if embedding_size is not None else self.target_embed_dimension
        )
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(
                self.forward_modules["feature_aggregator"].backbone.parameters(), lr
            )

        self.aed_meta_epochs = aed_meta_epochs

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(
                self.target_embed_dimension,
                self.target_embed_dimension,
                pre_proj,
                proj_layer_type,
            ).to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), lr * 0.1)

        # -------------------------
        # original SimpleNet noise discriminator
        # -------------------------
        self.use_noise_discriminator = bool(use_noise_discriminator)
        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.dsc_margin = dsc_margin

        self.noise_weight = float(noise_weight)
        self.noise_weight_mode = str(noise_weight_mode)
        self.noise_weight_min = float(noise_weight_min)
        self.noise_weight_decay_start = int(noise_weight_decay_start)
        self.noise_weight_decay_end = noise_weight_decay_end if noise_weight_decay_end is None else int(noise_weight_decay_end)
        self.noise_weight_off_after = noise_weight_off_after if noise_weight_off_after is None else int(noise_weight_off_after)

        # build even if disabled (사용자가 꺼두면 loss/step만 스킵)
        self.discriminator = Discriminator(
            self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden
        ).to(self.device)
        self.dsc_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5
        )
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.dsc_opt,
            (meta_epochs - aed_meta_epochs) * gan_epochs,
            self.dsc_lr * 0.4,
        )

        # -------------------------
        # pseudo heads
        # -------------------------
        self.use_pseudo_cls_head = bool(use_pseudo_cls_head)
        self.use_pseudo_seg_head = bool(use_pseudo_seg_head)

        self.pseudo_save_images = bool(pseudo_save_images)
        self.pseudo_save_every = max(1, int(pseudo_save_every))
        self.pseudo_save_max_per_epoch = int(pseudo_save_max_per_epoch)

        # cls schedule
        self.pseudo_cls_weight = float(pseudo_cls_weight)
        self.pseudo_cls_weight_mode = str(pseudo_cls_weight_mode)
        self.pseudo_cls_weight_min = float(pseudo_cls_weight_min)
        self.pseudo_cls_weight_decay_start = int(pseudo_cls_weight_decay_start)
        self.pseudo_cls_weight_decay_end = pseudo_cls_weight_decay_end if pseudo_cls_weight_decay_end is None else int(pseudo_cls_weight_decay_end)
        self.pseudo_cls_weight_off_after = pseudo_cls_weight_off_after if pseudo_cls_weight_off_after is None else int(pseudo_cls_weight_off_after)

        # seg schedule
        self.pseudo_seg_weight = float(pseudo_seg_weight)
        self.pseudo_seg_weight_mode = str(pseudo_seg_weight_mode)
        self.pseudo_seg_weight_min = float(pseudo_seg_weight_min)
        self.pseudo_seg_weight_decay_start = int(pseudo_seg_weight_decay_start)
        self.pseudo_seg_weight_decay_end = pseudo_seg_weight_decay_end if pseudo_seg_weight_decay_end is None else int(pseudo_seg_weight_decay_end)
        self.pseudo_seg_weight_off_after = pseudo_seg_weight_off_after if pseudo_seg_weight_off_after is None else int(pseudo_seg_weight_off_after)

        self.pseudo_cls_logit_topk = max(1, int(pseudo_cls_logit_topk))

        if (self.use_pseudo_cls_head or self.use_pseudo_seg_head) and pseudo_anomaly is None:
            raise ImportError(
                f"pseudo_anomaly.py import failed: {_PSEUDO_IMPORT_ERR}\n"
                f"Place pseudo_anomaly.py in importable path."
            )

        # cls head
        if self.use_pseudo_cls_head:
            self.pseudo_cls_discriminator = Discriminator(
                self.target_embed_dimension, n_layers=pseudo_cls_dsc_layers, hidden=pseudo_cls_dsc_hidden
            ).to(self.device)
            self.pseudo_cls_opt = torch.optim.Adam(
                self.pseudo_cls_discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5
            )
            self.pseudo_cls_criterion = torch.nn.BCEWithLogitsLoss()

        # seg head
        if self.use_pseudo_seg_head:
            self.pseudo_seg_discriminator = Discriminator(
                self.target_embed_dimension, n_layers=pseudo_seg_dsc_layers, hidden=pseudo_seg_dsc_hidden
            ).to(self.device)
            self.pseudo_seg_opt = torch.optim.Adam(
                self.pseudo_seg_discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5
            )
            self.pseudo_seg_criterion = torch.nn.BCEWithLogitsLoss()

        # bookkeeping
        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None
        self.pseudo_save_dir = None

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)

        # pseudo image save dir
        if self.use_pseudo_cls_head or self.use_pseudo_seg_head:
            self.pseudo_save_dir = os.path.join(self.ckpt_dir, "pseudo_anomaly_images")
            os.makedirs(self.pseudo_save_dir, exist_ok=True)

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                with torch.no_grad():
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        B = len(images)
        if (not evaluation) and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(
                    0, 3, 1, 2
                )

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])

            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        return features, patch_shapes

    def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt):
        scores = np.squeeze(np.array(scores))
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / (img_max_scores - img_min_scores + 1e-12)

        image_metrics = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt)
        auroc = image_metrics["auroc"]

        full_pixel_auroc = 0.0
        pro = 0.0
        return auroc, full_pixel_auroc, pro

    def train(self, training_data, test_data):
        state_dict = {}
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")

        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if "discriminator" in state_dict:
                self.discriminator.load_state_dict(state_dict["discriminator"])
                if self.pre_proj > 0 and "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
                if self.use_pseudo_cls_head and "pseudo_cls_discriminator" in state_dict:
                    self.pseudo_cls_discriminator.load_state_dict(state_dict["pseudo_cls_discriminator"])
                if self.use_pseudo_seg_head and "pseudo_seg_discriminator" in state_dict:
                    self.pseudo_seg_discriminator.load_state_dict(state_dict["pseudo_seg_discriminator"])
            else:
                self.load_state_dict(state_dict, strict=False)

            self.predict(training_data, "train_")
            scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            auroc, full_pixel_auroc, pro = self._evaluate(
                test_data, scores, segmentations, features, labels_gt, masks_gt
            )
            return auroc, full_pixel_auroc, pro

        def update_state_dict():
            state_dict["discriminator"] = OrderedDict(
                {k: v.detach().cpu() for k, v in self.discriminator.state_dict().items()}
            )
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict(
                    {k: v.detach().cpu() for k, v in self.pre_projection.state_dict().items()}
                )
            if self.use_pseudo_cls_head:
                state_dict["pseudo_cls_discriminator"] = OrderedDict(
                    {k: v.detach().cpu() for k, v in self.pseudo_cls_discriminator.state_dict().items()}
                )
            if self.use_pseudo_seg_head:
                state_dict["pseudo_seg_discriminator"] = OrderedDict(
                    {k: v.detach().cpu() for k, v in self.pseudo_seg_discriminator.state_dict().items()}
                )

        best_record = None
        for i_mepoch in range(self.meta_epochs):
            self._train_discriminator(training_data, meta_epoch=i_mepoch)

            scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            auroc, full_pixel_auroc, pro = self._evaluate(
                test_data, scores, segmentations, features, labels_gt, masks_gt
            )

            self.logger.logger.add_scalar("i-auroc", auroc, i_mepoch)
            self.logger.logger.add_scalar("p-auroc", full_pixel_auroc, i_mepoch)
            self.logger.logger.add_scalar("pro", pro, i_mepoch)

            if best_record is None:
                best_record = [auroc, full_pixel_auroc, pro]
                update_state_dict()
            else:
                if auroc > best_record[0]:
                    best_record = [auroc, full_pixel_auroc, pro]
                    update_state_dict()
                elif auroc == best_record[0] and full_pixel_auroc > best_record[1]:
                    best_record[1] = full_pixel_auroc
                    best_record[2] = pro
                    update_state_dict()

            print(
                f"----- {i_mepoch} I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)})"
                f"  P-AUROC{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) -----"
                f"  PRO-AUROC{round(pro, 4)}(MAX:{round(best_record[2], 4)}) -----"
            )

        torch.save(state_dict, ckpt_path)
        return best_record

    def _train_discriminator(self, input_data, meta_epoch: int = 0):
        _ = self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.train()

        # train modes
        self.discriminator.train()
        if self.use_pseudo_cls_head:
            self.pseudo_cls_discriminator.train()
        if self.use_pseudo_seg_head:
            self.pseudo_seg_discriminator.train()

        LOGGER.info("Training discriminator...")

        # -------------------------
        # scheduled weights per meta-epoch
        # -------------------------
        w_noise = self._schedule_weight(
            meta_epoch,
            enabled=self.use_noise_discriminator,
            w0=self.noise_weight,
            mode=self.noise_weight_mode,
            wmin=self.noise_weight_min,
            decay_start=self.noise_weight_decay_start,
            decay_end=self.noise_weight_decay_end,
            off_after=self.noise_weight_off_after,
        )
        w_cls = self._schedule_weight(
            meta_epoch,
            enabled=self.use_pseudo_cls_head,
            w0=self.pseudo_cls_weight,
            mode=self.pseudo_cls_weight_mode,
            wmin=self.pseudo_cls_weight_min,
            decay_start=self.pseudo_cls_weight_decay_start,
            decay_end=self.pseudo_cls_weight_decay_end,
            off_after=self.pseudo_cls_weight_off_after,
        )
        w_seg = self._schedule_weight(
            meta_epoch,
            enabled=self.use_pseudo_seg_head,
            w0=self.pseudo_seg_weight,
            mode=self.pseudo_seg_weight_mode,
            wmin=self.pseudo_seg_weight_min,
            decay_start=self.pseudo_seg_weight_decay_start,
            decay_end=self.pseudo_seg_weight_decay_end,
            off_after=self.pseudo_seg_weight_off_after,
        )

        # pseudo saving
        do_save_pseudo = (
            (self.use_pseudo_cls_head or self.use_pseudo_seg_head)
            and self.pseudo_save_images
            and (meta_epoch % self.pseudo_save_every == 0)
        )
        saved_this_epoch = 0

        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss_noise = []
                all_p_true = []
                all_p_fake = []

                all_loss_cls = []
                all_cls_acc_n = []
                all_cls_acc_p = []

                all_loss_seg = []
                all_seg_iou = []

                for bidx, data_item in enumerate(input_data):
                    # zero grad
                    if self.use_noise_discriminator and w_noise > 0.0:
                        self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    if self.train_backbone:
                        self.backbone_opt.zero_grad()
                    if self.use_pseudo_cls_head and w_cls > 0.0:
                        self.pseudo_cls_opt.zero_grad()
                    if self.use_pseudo_seg_head and w_seg > 0.0:
                        self.pseudo_seg_opt.zero_grad()

                    img = data_item["image"].to(torch.float).to(self.device)

                    # --- shared features (normal) ---
                    feats, patch_shapes = self._embed(img, evaluation=False)
                    if self.pre_proj > 0:
                        true_feats = self.pre_projection(feats)
                    else:
                        true_feats = feats

                    B = img.shape[0]
                    Hp, Wp = int(patch_shapes[0][0]), int(patch_shapes[0][1])
                    P = Hp * Wp

                    total_loss = torch.zeros([], device=self.device, dtype=true_feats.dtype)

                    # -------------------------
                    # 1) noise discriminator (original SimpleNet)
                    # -------------------------
                    disc_loss = None
                    p_true = None
                    p_fake = None

                    if self.use_noise_discriminator and w_noise > 0.0:
                        # true_feats is (B*P, D) => noise selection must be per-patch (N=B*P), not per-image (B)
                        N = true_feats.shape[0]  # == B*P
                        D = true_feats.shape[1]

                        noise_idxs = torch.randint(0, self.mix_noise, torch.Size([N]), device=self.device)  # (N,)
                        noise_one_hot = torch.nn.functional.one_hot(
                            noise_idxs, num_classes=self.mix_noise
                        ).to(self.device)  # (N, mix_noise)

                        # noise: (N, mix_noise, D)
                        noise = torch.stack(
                            [
                                torch.normal(
                                    0, self.noise_std * (1.1 ** (k)), (N, D), device=self.device
                                )
                                for k in range(self.mix_noise)
                            ],
                            dim=1,
                        )
                        # select mixture component per patch
                        noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)  # (N, D)
                        fake_feats = true_feats + noise

                        scores = self.discriminator(torch.cat([true_feats, fake_feats], dim=0))
                        true_scores = scores[: len(true_feats)]
                        fake_scores = scores[len(true_feats) :]

                        th = self.dsc_margin
                        p_true = (true_scores.detach() >= th).float().mean()
                        p_fake = (fake_scores.detach() < -th).float().mean()

                        true_loss = torch.clip(-true_scores + th, min=0)
                        fake_loss = torch.clip(fake_scores + th, min=0)
                        disc_loss = true_loss.mean() + fake_loss.mean()

                        total_loss = total_loss + (w_noise * disc_loss)

                    # -------------------------
                    # 2) generate pseudo once if any pseudo head active
                    # -------------------------
                    pseudo01 = None
                    pseudo_m01 = None
                    pseudo_feats = None

                    need_pseudo = ((self.use_pseudo_cls_head and w_cls > 0.0) or (self.use_pseudo_seg_head and w_seg > 0.0))
                    if need_pseudo:
                        img01 = _denorm_to_01(img)
                        seed = int(meta_epoch * 100000 + i_epoch * 1000 + bidx)

                        # seg head needs mask
                        if self.use_pseudo_seg_head and w_seg > 0.0:
                            pseudo01, pseudo_m01 = pseudo_anomaly.make_pseudo_batch(img01, seed=seed, return_mask=True)
                        else:
                            pseudo01 = pseudo_anomaly.make_pseudo_batch(img01, seed=seed, return_mask=False)
                            pseudo_m01 = None

                        # save pseudo images (and masks)
                        if do_save_pseudo:
                            epoch_dir = os.path.join(self.pseudo_save_dir, f"epoch_{meta_epoch:04d}")
                            for bi in range(pseudo01.shape[0]):
                                if self.pseudo_save_max_per_epoch > 0 and saved_this_epoch >= self.pseudo_save_max_per_epoch:
                                    break
                                out_path = os.path.join(epoch_dir, f"b{bidx:05d}_i{bi:02d}.png")
                                _save_tensor_img_01(pseudo01[bi], out_path)
                                if pseudo_m01 is not None:
                                    m_path = os.path.join(epoch_dir, f"b{bidx:05d}_i{bi:02d}_mask.png")
                                    _save_tensor_mask_01(pseudo_m01[bi], m_path)
                                saved_this_epoch += 1

                        pseudo_img = _norm_from_01(pseudo01)
                        pseudo_feats, _ = self._embed(pseudo_img, evaluation=False)
                        if self.pre_proj > 0:
                            pseudo_feats = self.pre_projection(pseudo_feats)

                    # -------------------------
                    # 3) pseudo classification head
                    # -------------------------
                    pseudo_cls_loss = None
                    cls_acc_n = None
                    cls_acc_p = None

                    if self.use_pseudo_cls_head and w_cls > 0.0:
                        patch_logit_n = self.pseudo_cls_discriminator(true_feats)      # (B*P,1)
                        patch_logit_p = self.pseudo_cls_discriminator(pseudo_feats)    # (B*P,1)

                        logit_n = self._patch_logits_to_image_logit(patch_logit_n, B, self.pseudo_cls_logit_topk)
                        logit_p = self._patch_logits_to_image_logit(patch_logit_p, B, self.pseudo_cls_logit_topk)

                        y0 = torch.zeros_like(logit_n)
                        y1 = torch.ones_like(logit_p)

                        pseudo_cls_loss = self.pseudo_cls_criterion(logit_n, y0) + self.pseudo_cls_criterion(logit_p, y1)

                        with torch.no_grad():
                            cls_acc_n = (logit_n <= 0.0).float().mean()
                            cls_acc_p = (logit_p > 0.0).float().mean()

                        total_loss = total_loss + (w_cls * pseudo_cls_loss)

                    # -------------------------
                    # 4) pseudo segmentation head (patch-grid BCE)
                    #   - normal image target: all zeros
                    #   - pseudo image target: downsample(pseudo_mask)
                    # -------------------------
                    pseudo_seg_loss = None
                    seg_iou = None

                    if self.use_pseudo_seg_head and w_seg > 0.0:
                        if pseudo_m01 is None:
                            raise RuntimeError("pseudo_seg_head enabled but pseudo mask was not returned. Check pseudo_anomaly.make_pseudo_batch(return_mask=True).")

                        # logits for normal & pseudo
                        logit_n = self.pseudo_seg_discriminator(true_feats).view(B, P)         # (B,P)
                        logit_p = self.pseudo_seg_discriminator(pseudo_feats).view(B, P)       # (B,P)

                        # downsample mask to patch grid
                        m01 = pseudo_m01.to(device=self.device, dtype=logit_p.dtype)  # (B,1,H,W)
                        m01p = self._downsample_mask_to_patch_grid(m01, (Hp, Wp))      # (B,1,Hp,Wp)
                        tgt_p = m01p.view(B, -1)                                       # (B,P)
                        tgt_n = torch.zeros_like(tgt_p)

                        # BCE on patches
                        loss_n = self.pseudo_seg_criterion(logit_n, tgt_n)
                        loss_p = self.pseudo_seg_criterion(logit_p, tgt_p)
                        pseudo_seg_loss = loss_n + loss_p
                        total_loss = total_loss + (w_seg * pseudo_seg_loss)

                        # quick IoU proxy (threshold=0.5 on sigmoid)
                        with torch.no_grad():
                            pred = (torch.sigmoid(logit_p) > 0.5).float()
                            gt = (tgt_p > 0.5).float()
                            inter = (pred * gt).sum(dim=1)
                            union = ((pred + gt) > 0).float().sum(dim=1).clamp(min=1.0)
                            seg_iou = (inter / union).mean()

                    # -------------------------
                    # logging (per step)
                    # -------------------------
                    if self.use_noise_discriminator and w_noise > 0.0 and disc_loss is not None:
                        self.logger.logger.add_scalar("noise/p_true", float(p_true.detach().cpu()), self.logger.g_iter)
                        self.logger.logger.add_scalar("noise/p_fake", float(p_fake.detach().cpu()), self.logger.g_iter)
                        self.logger.logger.add_scalar("noise/loss", float(disc_loss.detach().cpu()), self.logger.g_iter)
                    self.logger.logger.add_scalar("w/noise", float(w_noise), self.logger.g_iter)
                    self.logger.logger.add_scalar("w/pseudo_cls", float(w_cls), self.logger.g_iter)
                    self.logger.logger.add_scalar("w/pseudo_seg", float(w_seg), self.logger.g_iter)

                    if self.use_pseudo_cls_head and w_cls > 0.0 and pseudo_cls_loss is not None:
                        self.logger.logger.add_scalar("pseudo_cls/loss", float(pseudo_cls_loss.detach().cpu()), self.logger.g_iter)
                        self.logger.logger.add_scalar("pseudo_cls/acc_normal", float(cls_acc_n.detach().cpu()), self.logger.g_iter)
                        self.logger.logger.add_scalar("pseudo_cls/acc_pseudo", float(cls_acc_p.detach().cpu()), self.logger.g_iter)
                        self.logger.logger.add_scalar("pseudo_cls/topk", float(self.pseudo_cls_logit_topk), self.logger.g_iter)

                    if self.use_pseudo_seg_head and w_seg > 0.0 and pseudo_seg_loss is not None:
                        self.logger.logger.add_scalar("pseudo_seg/loss", float(pseudo_seg_loss.detach().cpu()), self.logger.g_iter)
                        if seg_iou is not None:
                            self.logger.logger.add_scalar("pseudo_seg/iou_proxy", float(seg_iou.detach().cpu()), self.logger.g_iter)

                    self.logger.step()

                    # -------------------------
                    # backward/step
                    # -------------------------
                    total_loss.backward()

                    if self.pre_proj > 0:
                        self.proj_opt.step()
                    if self.train_backbone:
                        self.backbone_opt.step()
                    if self.use_noise_discriminator and w_noise > 0.0:
                        self.dsc_opt.step()
                    if self.use_pseudo_cls_head and w_cls > 0.0:
                        self.pseudo_cls_opt.step()
                    if self.use_pseudo_seg_head and w_seg > 0.0:
                        self.pseudo_seg_opt.step()

                    # collect for pbar
                    if disc_loss is not None:
                        all_loss_noise.append(float(disc_loss.detach().cpu()))
                    if p_true is not None:
                        all_p_true.append(float(p_true.detach().cpu()))
                    if p_fake is not None:
                        all_p_fake.append(float(p_fake.detach().cpu()))

                    if pseudo_cls_loss is not None:
                        all_loss_cls.append(float(pseudo_cls_loss.detach().cpu()))
                        all_cls_acc_n.append(float(cls_acc_n.detach().cpu()))
                        all_cls_acc_p.append(float(cls_acc_p.detach().cpu()))

                    if pseudo_seg_loss is not None:
                        all_loss_seg.append(float(pseudo_seg_loss.detach().cpu()))
                        if seg_iou is not None:
                            all_seg_iou.append(float(seg_iou.detach().cpu()))

                if self.cos_lr and (self.use_noise_discriminator and w_noise > 0.0):
                    self.dsc_schl.step()

                cur_lr = self.dsc_opt.state_dict()["param_groups"][0]["lr"]

                # pbar string
                pbar_str = f"gan_epoch:{i_epoch} lr:{cur_lr:.6f} "
                pbar_str += f"w_noise:{w_noise:.3f} w_cls:{w_cls:.3f} w_seg:{w_seg:.3f} "

                if len(all_loss_noise) > 0:
                    pbar_str += f"noise_loss:{(sum(all_loss_noise)/len(all_loss_noise)):.5f} "
                    if len(all_p_true) > 0 and len(all_p_fake) > 0:
                        pbar_str += f"p_true:{(sum(all_p_true)/len(all_p_true)):.3f} "
                        pbar_str += f"p_fake:{(sum(all_p_fake)/len(all_p_fake)):.3f} "

                if len(all_loss_cls) > 0:
                    pbar_str += f"cls_loss:{(sum(all_loss_cls)/len(all_loss_cls)):.5f} "
                    pbar_str += f"cls_acc_n:{(sum(all_cls_acc_n)/len(all_cls_acc_n)):.3f} "
                    pbar_str += f"cls_acc_p:{(sum(all_cls_acc_p)/len(all_cls_acc_p)):.3f} "
                    pbar_str += f"cls_topk:{self.pseudo_cls_logit_topk:d} "

                if len(all_loss_seg) > 0:
                    pbar_str += f"seg_loss:{(sum(all_loss_seg)/len(all_loss_seg)):.5f} "
                    if len(all_seg_iou) > 0:
                        pbar_str += f"seg_iou:{(sum(all_seg_iou)/len(all_seg_iou)):.3f} "

                pbar.set_description_str(pbar_str)
                pbar.update(1)

    def predict(self, data, prefix=""):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, prefix):
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None:
                        masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                _scores, _masks, _feats = self._predict(image)
                for score, mask, feat in zip(_scores, _masks, _feats):
                    scores.append(score)
                    masks.append(mask)

        return scores, masks, features, labels_gt, masks_gt

    def _predict(self, images):
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()

        # inference: keep original SimpleNet discriminator (noise disc)
        self.discriminator.eval()

        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True, evaluation=True)
            if self.pre_proj > 0:
                features = self.pre_projection(features)

            patch_scores = image_scores = -self.discriminator(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            features = features.reshape(batchsize, scales[0], scales[1], -1)
            masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)

        return list(image_scores), list(masks), list(features)

    def save_segmentation_images(self, data, segmentations, scores):
        image_paths = [x[2] for x in data.dataset.data_to_iterate]
        mask_paths = [x[3] for x in data.dataset.data_to_iterate]

        def image_transform(image):
            in_std = np.array(data.dataset.transform_std).reshape(-1, 1, 1)
            in_mean = np.array(data.dataset.transform_mean).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip((image.numpy() * in_std + in_mean) * 255, 0, 255).astype(np.uint8)

        def mask_transform(mask):
            return data.dataset.transform_mask(mask).numpy()

        plot_segmentation_images(
            "./output",
            image_paths,
            segmentations,
            scores,
            mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )


class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)

        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))

        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x
