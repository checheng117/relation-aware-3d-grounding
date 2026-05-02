"""rag3d models module."""

from .sat_model import SATModel, build_sat_model
from .relation_aware_referit3dnet import RelationAwareReferIt3DNet, build_relation_aware_model

# COVER-3D modules
from .cover3d_wrapper import Cover3DWrapper, Cover3DInput, Cover3DOutput
from .cover3d_dense_relation import DenseRelationModule, ChunkedDenseRelationModule
from .cover3d_anchor_posterior import SoftAnchorPosteriorModule, AnchorPosteriorModule
from .cover3d_calibration import CalibratedFusionGate, FusionGate
from .cover3d_model import Cover3DModel, create_cover3d_model_from_config

__all__ = [
    'SATModel', 'build_sat_model',
    'RelationAwareReferIt3DNet', 'build_relation_aware_model',
    # COVER-3D
    'Cover3DWrapper', 'Cover3DInput', 'Cover3DOutput',
    'DenseRelationModule', 'ChunkedDenseRelationModule',
    'SoftAnchorPosteriorModule', 'AnchorPosteriorModule',
    'CalibratedFusionGate', 'FusionGate',
    'Cover3DModel', 'create_cover3d_model_from_config',
]