from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BINARY_MODEL_DIR = PROJECT_ROOT / "Binary classification model"
SUPERCLASS_MODEL_DIR = PROJECT_ROOT / "three-category superclass classification model"
SUBTYPE_MODEL_DIR = PROJECT_ROOT / "11-class ResNet50 subtype model"

BINARY_WEIGHTS = BINARY_MODEL_DIR / "best_auc_model_seed64.pth"
SUPERCLASS_WEIGHTS = SUPERCLASS_MODEL_DIR / "best_auc_model_seed53.pth"
SUBTYPE_WEIGHTS = SUBTYPE_MODEL_DIR / "best.pth"
DLIB_LANDMARK_MODEL = BINARY_MODEL_DIR / "shape_predictor_68_face_landmarks.dat"

BINARY_CLASSES = [
    "No scoliosis-related facial phenotype detected (healthy control)",
    "Scoliosis-related facial phenotype detected",
]
BINARY_DISEASE_INDEX = 1

SUPERCLASS_CLASSES = [
    "Syndromic scoliosis",
    "Chiari malformation-associated scoliosis",
    "Adolescent idiopathic scoliosis",
]
SUPERCLASS_SYNDROMIC_INDEX = 0

SS_SUBTYPE_CLASSES = [
    "Arthrogryposis multiplex congenita",
    "Ehlers-Danlos syndrome",
    "Freeman-Sheldon syndrome",
    "Gorham-Stout disease",
    "Marfan syndrome",
    "Neurofibromatosis type 1",
    "Osteochondrodysplasia",
    "Osteogenesis imperfecta",
    "Other syndromic scoliosis",
    "Prader-Willi syndrome",
    "Shprintzen-Goldberg syndrome",
]

SS_DISPLAY_THRESHOLD = 0.50

MODEL_VERSIONS = {
    "binary": BINARY_WEIGHTS.name,
    "etiology": SUPERCLASS_WEIGHTS.name,
    "ss_subtype": SUBTYPE_WEIGHTS.name,
}
