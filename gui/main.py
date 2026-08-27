"""NiceGUI entry: BEACON inference workbench."""

from __future__ import annotations

import threading
from pathlib import Path

from nicegui import app, ui

from .adapter import collect_image_paths, run_inference_job
from .defaults import DEFAULT_CHECKPOINTS, DEFAULT_OUTPUT_DIR, PROJECT_ROOT
from .state import config, run


_worker: threading.Thread | None = None
_last_gallery_n = -1
_last_error_notified = ""


def _start_job() -> None:
    global _worker, _last_gallery_n, _last_error_notified
    if run.status == "running":
        ui.notify("Job already running", type="warning")
        return
    try:
        collect_image_paths(config)
    except Exception as e:
        ui.notify(str(e), type="negative")
        run.log(f"Validation error: {e}")
        return
    if not config.resolved_targets():
        ui.notify("Select Tumor_EGFR and/or Mac_EREG", type="warning")
        return

    run.reset()
    _last_gallery_n = -1
    _last_error_notified = ""
    run.status = "running"
    run.log("Job queued…")
    _worker = threading.Thread(target=run_inference_job, args=(config, run), daemon=True)
    _worker.start()
    ui.notify("Inference started", type="positive")


def _refresh_results_gallery(container) -> None:
    container.clear()
    with container:
        if not run.results:
            ui.label("No results yet.").classes("text-grey")
            return
        ui.label(f"Run directory: {run.run_dir}").classes("text-sm break-all")
        if run.run_dir:
            summary = Path(run.run_dir) / "summary.csv"
            if summary.is_file():
                ui.button(
                    "Download summary.csv",
                    on_click=lambda p=str(summary): ui.download(p),
                ).props("dense flat")
        for r in run.results:
            with ui.card().classes("w-full"):
                ui.label(
                    f"{r['image_id']} · {r['target']}  |  "
                    f"mean={r['mean_abundance']:.4f}  patches={r['n_valid_patches']}"
                ).classes("font-medium")
                with ui.row().classes("w-full items-start gap-4 flex-wrap"):
                    if r.get("thumbnail") and Path(r["thumbnail"]).is_file():
                        ui.image(r["thumbnail"]).classes("w-40")
                    if r.get("heatmap_png") and Path(r["heatmap_png"]).is_file():
                        ui.image(r["heatmap_png"]).classes("w-80 max-w-full")
                with ui.row():
                    if r.get("heatmap_png") and Path(r["heatmap_png"]).is_file():
                        ui.button(
                            "Download PNG",
                            on_click=lambda p=r["heatmap_png"]: ui.download(p),
                        ).props("dense")
                    if r.get("heatmap_npy") and Path(r["heatmap_npy"]).is_file():
                        ui.button(
                            "Download NPY",
                            on_click=lambda p=r["heatmap_npy"]: ui.download(p),
                        ).props("dense")


def build_ui() -> None:
    ui.page_title("BEACON Inference Workbench")
    ui.colors(primary="#1f4e79")

    with ui.header().classes("items-center justify-between"):
        ui.label("BEACON · H&E → Tumor_EGFR / Mac_EREG").classes("text-h6")
        ui.label(f"root: {PROJECT_ROOT}").classes("text-xs opacity-70")

    with ui.row().classes("w-full items-stretch no-wrap gap-4 p-4"):
        with ui.column().classes("w-1/3 min-w-[320px] gap-3"):
            ui.label("Input").classes("text-h6")
            ui.markdown(
                "**Prefer a server path** to the H&E image or core folder. "
                "Do **not** upload multi-hundred-MB WSIs in the browser."
            ).classes("text-sm")

            ui.select(
                {"file": "Single image path", "directory": "Directory of cores"},
                label="Input mode",
                value=config.input_mode,
            ).bind_value(config, "input_mode").classes("w-full")

            ui.input("Image / directory path").bind_value(config, "input_path").classes(
                "w-full"
            ).props("dense outlined")
            ui.input("Output directory").bind_value(config, "output_dir").classes(
                "w-full"
            ).props("dense outlined")

            ui.separator()
            ui.label("Targets & checkpoints").classes("text-h6")
            ui.checkbox("Tumor_EGFR").bind_value(config, "run_tumor_egfr")
            ui.input("Tumor_EGFR weights").bind_value(config, "weights_tumor_egfr").classes(
                "w-full"
            ).props("dense outlined")
            ui.checkbox("Mac_EREG").bind_value(config, "run_mac_ereg")
            ui.input("Mac_EREG weights").bind_value(config, "weights_mac_ereg").classes(
                "w-full"
            ).props("dense outlined")
            ui.input("UNI weights (optional)").bind_value(config, "uni_weights_path").classes(
                "w-full"
            ).props("dense outlined")

            ui.separator()
            ui.label("Inference parameters").classes("text-h6")
            ui.select(
                ["auto", "cuda", "cpu"],
                label="Device",
                value=config.device,
            ).bind_value(config, "device").classes("w-full")
            with ui.row().classes("w-full gap-2"):
                ui.number("patch_size", format="%.0f").bind_value(config, "patch_size").classes(
                    "w-24"
                )
                ui.number("stride", format="%.0f").bind_value(config, "stride").classes("w-24")
                ui.number("batch_size", format="%.0f").bind_value(config, "batch_size").classes(
                    "w-24"
                )
            with ui.row().classes("w-full gap-2"):
                ui.number("k_neighbors", format="%.0f").bind_value(
                    config, "k_neighbors"
                ).classes("w-28")
                ui.number("mask_min_fraction", format="%.2f", step=0.05).bind_value(
                    config, "mask_min_fraction"
                ).classes("w-36")
            ui.checkbox("Save input thumbnail").bind_value(config, "make_thumbnail")

            ui.button("Run BEACON inference", on_click=_start_job).props(
                "color=primary unelevated"
            ).classes("w-full mt-2")

            def _restore_defaults() -> None:
                config.weights_tumor_egfr = str(DEFAULT_CHECKPOINTS["Tumor_EGFR"])
                config.weights_mac_ereg = str(DEFAULT_CHECKPOINTS["Mac_EREG"])
                config.output_dir = str(DEFAULT_OUTPUT_DIR)
                ui.notify("Defaults restored", type="info")

            ui.button("Reset checkpoint defaults", on_click=_restore_defaults).props("flat")

        with ui.column().classes("w-2/3 grow gap-3"):
            ui.label("Run status").classes("text-h6")
            status_label = ui.label().classes("text-sm")
            stage_label = ui.label().classes("text-sm text-grey")
            progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full")

            ui.label("Log").classes("text-subtitle1")
            log_area = (
                ui.textarea()
                .props("readonly outlined")
                .classes("w-full font-mono text-xs")
                .style("min-height: 220px")
            )

            ui.label("Results").classes("text-h6")
            results_box = ui.column().classes("w-full gap-2")

            def _tick() -> None:
                global _last_gallery_n, _last_error_notified
                status_label.set_text(f"Status: {run.status}")
                stage_label.set_text(f"Stage: {run.stage}" if run.stage else "")
                progress_bar.set_value(run.progress)
                log_area.value = "\n".join(run.logs[-400:])
                n = len(run.results)
                if n != _last_gallery_n and (run.status in {"done", "error", "running"}):
                    _last_gallery_n = n
                    _refresh_results_gallery(results_box)
                if run.status == "error" and run.error and run.error != _last_error_notified:
                    _last_error_notified = run.error
                    ui.notify(run.error, type="negative", timeout=8000)

            ui.timer(0.5, _tick)


def main() -> None:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app.add_media_files("/gui_runs", DEFAULT_OUTPUT_DIR)

    @ui.page("/")
    def index():
        build_ui()

    ui.run(
        title="BEACON Inference",
        host="0.0.0.0",
        port=8088,
        reload=False,
        show=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
