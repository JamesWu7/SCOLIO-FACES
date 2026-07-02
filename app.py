import html
import os
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from faces_inference import predict_faces


APP_ROOT = Path(__file__).resolve().parent

CLINICAL_NOTE = (
    "This tool is intended for research demonstration and clinical triage support within the "
    "SCOLIO-FACES framework. Model outputs are relative support scores and must not be used as "
    "standalone diagnoses. They do not replace standing whole-spine radiographs, spine magnetic "
    "resonance imaging, genetic testing, medical history, physical examination, or evaluation by "
    "spine surgery and medical genetics specialists. If the model suggests scoliosis-related or "
    "syndromic scoliosis-related support, standard clinical evaluation is still required."
)


def _to_dataframe(items, label_col="Class"):
    if not items:
        return pd.DataFrame([{label_col: "Not run", "Probability": "", "Percent": ""}])
    return pd.DataFrame(
        [
            {
                label_col: item["class_name"],
                "Probability": round(item["probability"], 3),
                "Percent": f"{item['probability_percent']:.1f}%",
            }
            for item in items
        ]
    )


def _plot_probabilities(items, title, limit=None):
    if not items:
        return None
    plot_items = items[:limit] if limit else items
    labels = [item["class_name"] for item in plot_items][::-1]
    values = [item["probability"] for item in plot_items][::-1]

    fig_height = max(3.2, 0.38 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    bars = ax.barh(range(len(labels)), values, color="#2A7F62", edgecolor="#1D4E40")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Model-derived relative support")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            min(value + 0.02, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left" if value < 0.92 else "right",
            fontsize=9,
        )
    fig.tight_layout()
    return fig


def _summary_markdown(result):
    top = result["top_summary"]
    if result.get("downstream_skipped"):
        downstream_block = (
            "- Etiology superclass: `Not run`\n"
            "- Syndromic scoliosis subtype ranking: `Not run`\n\n"
            "The binary screening result did not suggest a scoliosis-related facial phenotype, "
            "so the etiology superclass model and the syndromic scoliosis subtype model were not run."
        )
    else:
        subtype_lines = "\n".join(
            [
                f"- {item['class_name']}: {item['probability']:.3f}"
                for item in top["ss_subtype_top3"]
            ]
        )
        downstream_block = (
            f"- Etiology superclass with highest support: `{top['etiology_top1']}` ({top['etiology_probability']:.3f})\n"
            f"- Top three syndromic scoliosis subtype supports:\n{subtype_lines}"
        )
    warnings = "\n".join([f"- {html.escape(w)}" for w in result["warnings"]])
    warning_block = f"\n\n**Notes**\n{warnings}" if warnings else ""

    return f"""
**Hierarchical prediction summary**

- Binary screening class with highest support: `{top["screening_result"]}` ({top["screening_probability"]:.3f})
{downstream_block}

**Run information**

- Face crop method: `{result["face_crop_method"]}`
- Inference device: `{result["device"]}`
{warning_block}
"""


def run_faces_app(image: Image.Image):
    try:
        result = predict_faces(image)
    except Exception as exc:
        message = f"Inference could not be completed: {exc}"
        empty = pd.DataFrame(columns=["Class", "Probability", "Percent"])
        return None, message, empty, empty, empty, None, CLINICAL_NOTE

    binary_df = _to_dataframe(result["binary"])
    etiology_df = _to_dataframe(result["etiology"])
    subtype_df = _to_dataframe(result["ss_subtypes"])
    fig = _plot_probabilities(result["ss_subtypes"], "Ranked syndromic scoliosis subtype support")

    return (
        result["cropped_face"],
        _summary_markdown(result),
        binary_df,
        etiology_df,
        subtype_df,
        fig,
        CLINICAL_NOTE,
    )


with gr.Blocks(title="SCOLIO-FACES", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# SCOLIO-FACES

Upload one clear, unobstructed frontal facial photograph. The system first performs binary screening for scoliosis-related facial phenotype support. Only when the binary model suggests a scoliosis-related facial phenotype will the app continue to the etiology superclass model and the eleven-class syndromic scoliosis subtype model.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload frontal face photograph")
            submit_btn = gr.Button("Run hierarchical prediction", variant="primary")
        with gr.Column(scale=1):
            cropped_output = gr.Image(type="pil", label="Cropped face preview", interactive=False)
            summary_output = gr.Markdown(label="Hierarchical prediction summary")

    with gr.Tabs():
        with gr.Tab("Binary screening"):
            binary_table = gr.Dataframe(label="Binary screening probabilities", interactive=False)
        with gr.Tab("Etiology superclass"):
            etiology_table = gr.Dataframe(label="Etiology superclass probabilities", interactive=False)
        with gr.Tab("Syndromic subtype ranking"):
            subtype_table = gr.Dataframe(label="Syndromic scoliosis subtype probabilities", interactive=False)
            subtype_plot = gr.Plot(label="Ranked syndromic scoliosis subtype support")
        with gr.Tab("Clinical note"):
            clinical_note = gr.Textbox(
                value=CLINICAL_NOTE,
                label="Clinical note",
                lines=5,
                interactive=False,
            )

    submit_btn.click(
        fn=run_faces_app,
        inputs=image_input,
        outputs=[
            cropped_output,
            summary_output,
            binary_table,
            etiology_table,
            subtype_table,
            subtype_plot,
            clinical_note,
        ],
    )


if __name__ == "__main__":
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port, show_api=False)
