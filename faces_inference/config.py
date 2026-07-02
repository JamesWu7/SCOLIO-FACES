from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BINARY_MODEL_DIR = PROJECT_ROOT / "Binary classification model"
SUPERCLASS_MODEL_DIR = PROJECT_ROOT / "three-category superclass classification model"
SUBTYPE_MODEL_DIR = PROJECT_ROOT / "11-class ResNet50 subtype model"

BINARY_WEIGHTS = BINARY_MODEL_DIR / "best_auc_model_seed64.pth"
SUPERCLASS_WEIGHTS = SUPERCLASS_MODEL_DIR / "best_auc_model_seed53.pth"
SUBTYPE_WEIGHTS = SUBTYPE_MODEL_DIR / "best.pth"
DLIB_LANDMARK_MODEL = BINARY_MODEL_DIR / "shape_predictor_68_face_landmarks.dat"

BINARY_CLASSES = ["未提示脊柱侧弯相关面部特征（健康对照）", "提示脊柱侧弯相关面部特征"]
BINARY_DISEASE_INDEX = 1

SUPERCLASS_CLASSES = [
    "综合征型脊柱侧弯",
    "Chiari 畸形相关脊柱侧弯",
    "青少年特发性脊柱侧弯",
]
SUPERCLASS_SYNDROMIC_INDEX = 0

SS_SUBTYPE_CLASSES = [
    "多发性关节挛缩症",
    "埃勒斯-当洛斯综合征",
    "弗里曼-谢尔登综合征",
    "戈勒姆-斯托特病",
    "马方综合征",
    "1型神经纤维瘤病",
    "骨软骨发育不良",
    "成骨不全",
    "其他综合征",
    "普拉德-威利综合征",
    "施普林岑-戈德堡综合征",
]

SS_DISPLAY_THRESHOLD = 0.50

MODEL_VERSIONS = {
    "binary": BINARY_WEIGHTS.name,
    "etiology": SUPERCLASS_WEIGHTS.name,
    "ss_subtype": SUBTYPE_WEIGHTS.name,
}
