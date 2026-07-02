---
title: 面部特征辅助脊柱侧弯病因分层
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
license: apache-2.0
---

<p align="center">
  <img src="logo.png" alt="Faces Logo" width="500"/>
</p>

# 面部特征辅助脊柱侧弯病因分层

基于正面人脸照片的脊柱侧弯病因分层科研演示工具。

本项目使用深度学习模型，对儿童和青少年脊柱侧弯进行分层筛查和病因支持度排序。模型输出仅表示相对支持度，不是独立诊断结论。

## 网页应用和部署

本仓库包含一个 Gradio 网页应用，推理流程为：

1. 二分类筛查：未提示脊柱侧弯相关面部特征（健康对照） versus 提示脊柱侧弯相关面部特征。
2. 若二分类提示脊柱侧弯相关面部特征，再进行病因大类分层：青少年特发性脊柱侧弯、Chiari 畸形相关脊柱侧弯、综合征型脊柱侧弯。
3. 若进入后续流程，输出综合征型脊柱侧弯 11 类细分支持度排序。

本应用仅用于科研展示和临床分诊辅助。标准临床评估、站立位全脊柱放射片、脊柱磁共振成像、遗传检测和专科医生评估仍然是必要环节。

## Option 1: Local installation

```bash
cd /Users/jameswoo/Desktop/FACES
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL shown in the terminal, usually:

```text
http://127.0.0.1:7860
```

Optional: install `dlib` if 68-point landmark cropping is required in your environment. If `dlib` is unavailable, the app falls back to OpenCV frontal-face detection.

```bash
pip install dlib
```

After dependency installation, run a CPU smoke test:

```bash
python tests/smoke_test.py
```

For hospital intranet deployment, change `server_name` in `app.py` from `127.0.0.1` to the internal host address or `0.0.0.0`, and deploy only behind an approved internal network boundary.

## Option 2: Hugging Face Spaces deployment

Hugging Face Spaces can run the real PyTorch/Gradio inference app.

1. Create a new Space at Hugging Face.
2. Select `Gradio` as the SDK.
3. Upload or push this repository, including:
   - `app.py`
   - `faces_inference/`
   - `requirements.txt`
   - the three `.pth` checkpoint files
   - `logo.png`
4. Wait for the Space build to finish.
5. Test a non-face image first, then a consented frontal face image.

If the push fails because checkpoint files are large, enable Git LFS for `*.pth` before pushing to Hugging Face:

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

## Option 3: GitHub Pages documentation

GitHub Pages cannot run PyTorch inference. It should be used only as a static project page.

The static documentation page is in:

```text
docs/index.html
```

To publish it:

1. Push the repository to GitHub.
2. Open repository `Settings`.
3. Go to `Pages`.
4. Choose `Deploy from a branch`.
5. Select the branch and `/docs` folder.
6. Save and wait for the Pages URL.

## RShiny note

RShiny is feasible, but it is not used in v1. The model stack is already Python/PyTorch, so a Shiny app would need either `reticulate` to call `faces_inference.predict_faces()` or a separate Shiny UI that calls a Python API. That adds cross-language deployment complexity without improving the first public demo.

## Privacy and data governance

- The app performs single-session inference and does not intentionally save uploaded raw facial images.
- Do not commit raw facial images, individual-level clinical data or patient identifiers to GitHub.
- Facial photographs are identifiable medical data. Use explicit consent, restricted access and local institutional governance before any clinical or external use.
- Public demo deployments should use consented demonstration images only. Avoid uploading patient faces to public services unless institutional approval and consent explicitly cover that use.

## Model files

The current app loads the existing local checkpoints:

- `Binary classification model/best_auc_model_seed64.pth`
- `three-category superclass classification model/best_auc_model_seed53.pth`
- `11-class ResNet50 subtype model/best.pth`

If these files are moved, update `faces_inference/config.py`.
