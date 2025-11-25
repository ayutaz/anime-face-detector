from __future__ import annotations

import pathlib
import warnings

import cv2
import numpy as np
import torch.nn as nn
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config
from mmengine.registry import DefaultScope
from mmpose.apis import inference_topdown, init_model


class LandmarkDetector:
    def __init__(
        self,
        landmark_detector_config_or_path: Config | str | pathlib.Path,
        landmark_detector_checkpoint_path: str | pathlib.Path,
        face_detector_config_or_path: Config | str | pathlib.Path | None = None,
        face_detector_checkpoint_path: str | pathlib.Path | None = None,
        device: str = 'cuda:0',
        flip_test: bool = True,
        box_scale_factor: float = 1.1,
    ):
        landmark_config = self._load_config(landmark_detector_config_or_path)
        face_detector_config = self._load_config(face_detector_config_or_path)

        self.landmark_detector = self._init_pose_model(
            landmark_config, landmark_detector_checkpoint_path, device, flip_test
        )
        self.face_detector = self._init_face_detector(
            face_detector_config, face_detector_checkpoint_path, device
        )

        self.box_scale_factor = box_scale_factor

    @staticmethod
    def _load_config(
        config_or_path: Config | str | pathlib.Path | None,
    ) -> Config | None:
        if config_or_path is None or isinstance(config_or_path, Config):
            return config_or_path
        return Config.fromfile(config_or_path)

    @staticmethod
    def _init_pose_model(
        config: Config,
        checkpoint_path: str | pathlib.Path,
        device: str,
        flip_test: bool,
    ) -> nn.Module:
        if isinstance(checkpoint_path, pathlib.Path):
            checkpoint_path = checkpoint_path.as_posix()
        model = init_model(config, checkpoint_path, device=device)

        # Set flip_test in model's test_cfg
        if hasattr(model, 'test_cfg') and model.test_cfg is not None:
            model.test_cfg['flip_test'] = flip_test
        if hasattr(model.cfg, 'model') and hasattr(model.cfg.model, 'test_cfg'):
            model.cfg.model.test_cfg['flip_test'] = flip_test

        # Set dataset_meta with our custom keypoint info (28 keypoints for anime face)
        if hasattr(config, 'dataset_info'):
            dataset_meta = {
                'dataset_name': config.dataset_info.get('dataset_name', 'anime_face'),
                'num_keypoints': 28,
                'keypoint_info': config.dataset_info.get('keypoint_info', {}),
                'skeleton_info': config.dataset_info.get('skeleton_info', {}),
                'joint_weights': config.dataset_info.get('joint_weights', [1.0] * 28),
                'sigmas': config.dataset_info.get('sigmas', [0.025] * 28),
                'flip_indices': config.dataset_info.get(
                    'flip_indices', config.flip_indices if hasattr(config, 'flip_indices') else []
                ),
            }
            model.dataset_meta = dataset_meta

        # Copy all config attributes to model.cfg (required for inference_topdown)
        for key in ['test_dataloader', 'test_pipeline', 'codec', 'flip_indices']:
            if hasattr(config, key) and not hasattr(model.cfg, key):
                setattr(model.cfg, key, getattr(config, key))
        return model

    @staticmethod
    def _init_face_detector(
        config: Config | None, checkpoint_path: str | pathlib.Path | None, device: str
    ) -> nn.Module | None:
        if config is not None:
            if isinstance(checkpoint_path, pathlib.Path):
                checkpoint_path = checkpoint_path.as_posix()
            model = init_detector(config, checkpoint_path, device=device)
        else:
            model = None
        return model

    def _detect_faces(self, image: np.ndarray) -> list[np.ndarray]:
        # Set mmdet scope for face detection
        with DefaultScope.overwrite_default_scope('mmdet'):
            # mmdet 3.x returns DetDataSample
            result = inference_detector(self.face_detector, image)
        # Extract bboxes and scores from pred_instances
        pred_instances = result.pred_instances
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        # Combine to [x0, y0, x1, y1, score] format
        boxes = []
        for bbox, score in zip(bboxes, scores):
            box = np.append(bbox, score)
            boxes.append(box)
        # scale boxes by `self.box_scale_factor`
        boxes = self._update_pred_box(boxes)
        return boxes

    def _update_pred_box(self, pred_boxes: np.ndarray) -> list[np.ndarray]:
        boxes = []
        for pred_box in pred_boxes:
            box = pred_box[:4]
            size = box[2:] - box[:2] + 1
            new_size = size * self.box_scale_factor
            center = (box[:2] + box[2:]) / 2
            tl = center - new_size / 2
            br = tl + new_size
            pred_box[:4] = np.concatenate([tl, br])
            boxes.append(pred_box)
        return boxes

    def _detect_landmarks(
        self, image: np.ndarray, boxes: list[np.ndarray]
    ) -> list[dict[str, np.ndarray]]:
        # mmpose 1.x uses inference_topdown with different interface
        # Convert boxes to numpy array format expected by inference_topdown
        bboxes = np.array(boxes) if boxes else np.empty((0, 5))

        # Set mmpose scope for landmark detection
        with DefaultScope.overwrite_default_scope('mmpose'):
            # inference_topdown returns list of PoseDataSample
            # Pass only first 4 columns (x0, y0, x1, y1) - mmpose 1.x expects (N, 4) format
            results = inference_topdown(
                self.landmark_detector, image, bboxes[:, :4], bbox_format='xyxy'
            )

        # Convert PoseDataSample to dict format for backward compatibility
        preds = []
        for i, result in enumerate(results):
            pred_instances = result.pred_instances
            keypoints = pred_instances.keypoints[0]  # (K, 2)
            keypoint_scores = pred_instances.keypoint_scores[0]  # (K,)
            # Combine keypoints and scores to [x, y, score] format
            keypoints_with_scores = np.concatenate(
                [keypoints, keypoint_scores[:, np.newaxis]], axis=1
            )
            preds.append({'bbox': boxes[i], 'keypoints': keypoints_with_scores})
        return preds

    @staticmethod
    def _load_image(image_or_path: np.ndarray | str | pathlib.Path) -> np.ndarray:
        if isinstance(image_or_path, np.ndarray):
            image = image_or_path
        elif isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
        elif isinstance(image_or_path, pathlib.Path):
            image = cv2.imread(image_or_path.as_posix())
        else:
            raise ValueError
        return image

    def __call__(
        self,
        image_or_path: np.ndarray | str | pathlib.Path,
        boxes: list[np.ndarray] | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Detect face landmarks.

        Args:
            image_or_path: An image with BGR channel order or an image path.
            boxes: A list of bounding boxes for faces. Each bounding box
                should be of the form [x0, y0, x1, y1, [score]].

        Returns: A list of detection results. Each detection result has
            bounding box of the form [x0, y0, x1, y1, [score]], and landmarks
            of the form [x, y, score].
        """
        image = self._load_image(image_or_path)
        if boxes is None:
            if self.face_detector is not None:
                boxes = self._detect_faces(image)
            else:
                warnings.warn(
                    'Neither the face detector nor the bounding box is '
                    'specified. So the entire image is treated as the face '
                    'region.'
                )
                h, w = image.shape[:2]
                boxes = [np.array([0, 0, w - 1, h - 1, 1])]
        return self._detect_landmarks(image, boxes)
