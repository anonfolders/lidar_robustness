from enum import Enum, auto

class PerturbationMode(Enum):
    GLOBAL_UNIFORM = auto()
    GLOBAL_NORMAL = auto()
    GLOBAL_LAPLACE = auto()

    LOCAL_UNIFORM = auto()
    LOCAL_NORMAL = auto()
    LOCAL_LAPLACE = auto()

    DIRECT_UNIFORM = auto()
    DIRECT_NORMAL = auto()
    DIRECT_LAPLACE = auto()

    GLOBAL_FALSE_POSITIVES = auto()
    LOCAL_FALSE_POSITIVES = auto()

    REFLECT_DOWNSAMPLE = auto()
    REFLECT_UPSAMPLE = auto()

    REFLECT_OTHER = auto()


class LocalDirection(Enum):
    X_PLUS = auto()
    X_MINUS = auto()
    Y_PLUS = auto()
    Y_MINUS = auto()
    Z_PLUS = auto()
    Z_MINUS = auto()


VALIDATION_FRAMES = 'validation_frames.txt'
POINTS_IN_BBOXES_FILENAME = 'projected_bboxes.txt'
KITTI_VELODYNE = '<../KITTI_DATASET/training/velodyne>'
RESULTS_FOLDER = '<>'
BASELINE_PCD_FOLDER = '<>'
