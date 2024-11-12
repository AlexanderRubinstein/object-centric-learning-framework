# Copied from https://raw.githubusercontent.com/qqlu/Entity/main/Entityv2/CropFormer/demo_cropformer/demo_from_dirs.py
from PIL import Image
import copy
import torch
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from get_segments.utils import add_maskformer2_config
from tqdm import tqdm
import get_segments.cropformer_model
from get_segments.tardataset import TarDataset
from get_segments.utils import BatchResizeShortestEdge, EntityCrop, EntityCropTransform
import detectron2.data.transforms as T


# def in_range(i, range):
#     if range is None:
#         return True
#     else:
#         start, end = range[0], range[1]
#         return start <= i < end

class EntityNetV2(DefaultPredictor):
    def __init__(self, args, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        super().__init__(cfg)
        self.model = self.model.cuda()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
        self.instance_mode = instance_mode

        # Get dataset loaded
        augs, crop_augs = self.generate_img_augs(cfg)
        dataset = TarDataset(args.input, transforms_help=[augs, crop_augs])
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False, collate_fn=None, pin_memory=True)
        self.confidence_threshold = args.confidence_threshold


    def generate_img_augs(self, cfg):
        shortest_side = np.random.choice([cfg.INPUT.MIN_SIZE_TEST])
        augs = [T.ResizeShortestEdge((shortest_side,), cfg.INPUT.MAX_SIZE_TEST, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,),]

        # Build original image augmentation
        crop_augs = []
        entity_crops = EntityCrop(cfg.ENTITY.CROP_AREA_RATIO,
                                    cfg.ENTITY.CROP_STRIDE_RATIO,
                                    cfg.ENTITY.CROP_SAMPLE_NUM_TEST,
                                    False)
        crop_augs.append(entity_crops)

        entity_resize = BatchResizeShortestEdge((shortest_side,), cfg.INPUT.MAX_SIZE_TEST, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)
        crop_augs.append(entity_resize)

        crop_augs = T.AugmentationList(crop_augs)
        return augs, crop_augs


    def run(self, range=None):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        mega_dict = {}
        with torch.inference_mode():
            for i, (inputs, filename) in enumerate(tqdm(self.dataloader)):
                if range is not None:
                    if i >= range[1]:
                        break
                    if i < range[0]:
                        continue
                filename = filename[0]
                inputs['image'], inputs['image_crop'] = inputs['image'].squeeze(0).cuda(non_blocking=True), inputs['image_crop'].squeeze(0).cuda(non_blocking=True)
                predictions = self.model([inputs])[0]
                pred_masks = predictions["instances"].pred_masks
                pred_scores = predictions["instances"].scores
                selected_indexes = (pred_scores >= self.confidence_threshold)
                selected_scores = pred_scores[selected_indexes]
                selected_masks  = pred_masks[selected_indexes]
                _, m_H, m_W = selected_masks.shape
                mask_id = np.zeros((m_H, m_W), dtype=np.uint8)

                selected_scores, ranks = torch.sort(selected_scores)
                ranks = ranks + 1

                for index in ranks:
                    mask_id[(selected_masks[index-1]==1).cpu().numpy()] = int(index)

                mega_dict[filename] = {"mask": mask_id, "scores": selected_scores}
        return mega_dict


