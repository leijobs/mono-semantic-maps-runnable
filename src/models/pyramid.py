import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidOccupancyNetwork(nn.Module):

    def __init__(self, frontend, transformer, topdown, classifier):
        super().__init__()

        self.frontend = frontend
        self.transformer = transformer
        self.topdown = topdown
        self.classifier = classifier

    def forward(self, image, calib, *args):
        # Extract multiscale feature maps
        feature_maps = self.frontend(image)

        # Transform image features to birds-eye-view
        bev_feats = self.transformer(feature_maps, calib)

        # Apply topdown network
        td_feats = self.topdown(bev_feats)

        # Predict individual class log-probabilities
        logits = self.classifier(td_feats)
        return logits


class PyramidFusionOccupancyNetwork(nn.Module):

    def __init__(self, frontend, point_frontend, transformer, fusion_topdown, topdown, classifier):
        super().__init__()

        self.frontend = frontend
        self.point_frontend = point_frontend
        self.transformer = transformer
        self.fusion_topdown = fusion_topdown
        self.topdown = topdown
        self.classifier = classifier

    def forward(self, image, point, calib, *args):
        # Extract multiscale feature maps
        feature_maps = self.frontend(image)

        # Extract multiscale feature maps
        point_feature_maps = self.point_frontend(point)

        # Transform image features to birds-eye-view
        bev_feats = self.transformer(feature_maps, calib)

        # Apply topdown network
        td_feats = self.topdown(bev_feats)

        # Apply topdown network
        td_feats = self.fusion_topdown(bev_feats, point_feature_maps)

        # Predict individual class log-probabilities
        logits = self.classifier(td_feats)
        return logits


class PyramidFusionDetectionNetwork(nn.Module):

    def __init__(self, frontend, point_frontend, transformer, fusion_topdown, detector):
        super().__init__()

        self.frontend = frontend
        self.point_frontend = point_frontend
        self.transformer = transformer
        self.fusion_topdown = fusion_topdown
        self.detector = detector

    def forward(self, image, point, calib, *args):
        # Extract multiscale feature maps
        feature_maps = self.frontend(image)

        # Extract multiscale feature maps
        point_feature_maps = self.point_frontend(point)

        # Transform image features to birds-eye-view
        bev_feats = self.transformer(feature_maps, calib)

        # Apply topdown network
        td_feats = self.fusion_topdown(bev_feats, point_feature_maps)

        # Predict individual class log-probabilities
        logits = self.detector(td_feats)
        return logits