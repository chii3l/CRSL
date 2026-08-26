import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PPG_COMPONENTS = ["clean_y_hat", "baseline_hat", "noise_hat", "y_hat"]
SPECTRAL_COMPONENTS = [
    "background_spectrum",
    "primary_target_spectrum",
    "target_glucose_spectrum",
    "target_spo2_spectrum",
    "target_spectrum",
    "spectrum",
]
SPECTRAL_CMAPS = {
    "background_spectrum": "Greens",
    "primary_target_spectrum": "Reds",
    "target_glucose_spectrum": "Reds",
    "target_spo2_spectrum": "Blues",
    "target_spectrum": "Oranges",
    "spectrum": "viridis",
}
SPECTRAL_COLORS = {
    "background_spectrum": "#2b9348",
    "primary_target_spectrum": "#d94b4b",
    "target_glucose_spectrum": "#d94b4b",
    "target_spo2_spectrum": "#2f80ed",
    "target_spectrum": "#f08a42",
    "spectrum": "#264653",
}


def _spectral_component_keys(data):
    keys = [key for key in SPECTRAL_COMPONENTS if key in data]
    if (
            "primary_target_spectrum" in keys and
            "target_glucose_spectrum" in keys):
        keys.remove("target_glucose_spectrum")
    return keys


def _load_component(data, key, sample_index):
    if key not in data:
        return None
    arr = np.asarray(data[key])
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        sample_index = min(sample_index, arr.shape[0] - 1)
        return arr[sample_index]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"{key} must have shape [N,C,T] or [C,T], got {arr.shape}")


def _load_spectrum(data, key, sample_index):
    if key not in data:
        return None
    arr = np.asarray(data[key])
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        sample_index = min(sample_index, arr.shape[0] - 1)
        return arr[sample_index]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"{key} must have shape [N,L,T] or [L,T], got {arr.shape}")


def _parse_channels(channels, n_channels, max_channels):
    if channels is None or str(channels).lower() == "all":
        return list(range(min(n_channels, max_channels)))
    selected = []
    for item in str(channels).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0 or value >= n_channels:
            raise ValueError(f"channel index {value} is out of range for {n_channels} channels")
        selected.append(value)
    return selected[:max_channels]


def _time_axis(n_steps, sampling_rate):
    if sampling_rate and sampling_rate > 0:
        return np.arange(n_steps) / float(sampling_rate), "Time (s)"
    return np.arange(n_steps), "Time index"


def _save_observation_domain(data, out_dir, sample_index, channels, sampling_rate, dpi):
    y_real = _load_component(data, "y_real", sample_index)
    y_hat = _load_component(data, "y_hat", sample_index)
    clean_y_hat = _load_component(data, "clean_y_hat", sample_index)
    baseline_hat = _load_component(data, "baseline_hat", sample_index)
    noise_hat = _load_component(data, "noise_hat", sample_index)

    first = next(x for x in [y_real, y_hat, clean_y_hat, baseline_hat, noise_hat] if x is not None)
    n_channels, n_steps = first.shape
    selected = _parse_channels(channels, n_channels, max_channels=6)
    t, xlabel = _time_axis(n_steps, sampling_rate)

    fig, axes = plt.subplots(
        len(selected),
        4,
        figsize=(18, max(2.2 * len(selected), 4.0)),
        squeeze=False,
        sharex=True)
    col_titles = [
        "Observed / reconstructed PPG",
        "Clean optical response",
        "Baseline drift",
        "High-frequency artifact",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row, ch in enumerate(selected):
        ax = axes[row, 0]
        if y_real is not None:
            ax.plot(t, y_real[ch], color="#23364f", linewidth=1.2, label="observed PPG")
        if y_hat is not None:
            ax.plot(t, y_hat[ch], color="#d94b4b", linewidth=1.0, alpha=0.9, label="y_hat")
        ax.legend(loc="upper right", fontsize=7, frameon=False)

        if clean_y_hat is not None:
            axes[row, 1].plot(t, clean_y_hat[ch], color="#2a9d8f", linewidth=1.1)
        if baseline_hat is not None:
            axes[row, 2].plot(t, baseline_hat[ch], color="#f08a42", linewidth=1.1)
        if noise_hat is not None:
            axes[row, 3].plot(t, noise_hat[ch], color="#6f55b5", linewidth=0.9)

        axes[row, 0].set_ylabel(f"Ch {ch}", fontsize=10, fontweight="bold")
        for col in range(4):
            axes[row, col].grid(True, alpha=0.25, linewidth=0.5)

    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel)

    fig.suptitle(
        f"Observation-domain H decomposition, sample {sample_index}",
        fontsize=14,
        fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / f"sample_{sample_index:04d}_observation_domain_h_components.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_heatmap(ax, image, wavelengths, time_values, title, cmap="viridis"):
    extent = [time_values[0], time_values[-1], wavelengths[0], wavelengths[-1]]
    im = ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Wavelength (nm)")
    ax.grid(False)
    return im


def _save_projected_spectral_domain(data, out_dir, sample_index, sampling_rate, dpi):
    if "H" not in data:
        return None
    H = np.asarray(data["H"], dtype=np.float64)
    if H.ndim != 2:
        raise ValueError(f"H must have shape [C,L], got {H.shape}")

    components = {key: _load_component(data, key, sample_index) for key in PPG_COMPONENTS}
    components = {key: value for key, value in components.items() if value is not None}
    if not components:
        return None

    n_channels, n_steps = next(iter(components.values())).shape
    if H.shape[0] != n_channels:
        raise ValueError(
            f"H channel count {H.shape[0]} does not match component channels {n_channels}")

    wavelengths = np.asarray(data["wavelengths"]) if "wavelengths" in data else np.arange(H.shape[1])
    H_pinv = np.linalg.pinv(H)
    projected = {
        key: H_pinv @ value.astype(np.float64)
        for key, value in components.items()
    }
    t, xlabel = _time_axis(n_steps, sampling_rate)

    order = [key for key in PPG_COMPONENTS if key in projected]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), squeeze=False, sharex=True, sharey=True)
    cmaps = {
        "clean_y_hat": "viridis",
        "baseline_hat": "YlOrBr",
        "noise_hat": "Purples",
        "y_hat": "magma",
    }
    for ax, key in zip(axes.ravel(), order):
        im = _plot_heatmap(
            ax,
            projected[key],
            wavelengths,
            t,
            f"{key} projected to spectral domain",
            cmap=cmaps.get(key, "viridis"))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel)
    fig.suptitle(
        "Spectral-domain projection by pseudo-inverse of H "
        "(interpretive, not a native model output)",
        fontsize=13,
        fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = out_dir / f"sample_{sample_index:04d}_spectral_projected_h_components.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _save_native_spectral_domain(data, out_dir, sample_index, sampling_rate, dpi):
    component_keys = _spectral_component_keys(data)
    spectra = {
        key: _load_spectrum(data, key, sample_index)
        for key in component_keys
    }
    spectra = {key: value for key, value in spectra.items() if value is not None}
    if not spectra:
        return None

    n_steps = next(iter(spectra.values())).shape[-1]
    wavelengths = (
        np.asarray(data["wavelengths"])
        if "wavelengths" in data
        else np.arange(next(iter(spectra.values())).shape[0]))
    t, xlabel = _time_axis(n_steps, sampling_rate)
    order = [key for key in component_keys if key in spectra]

    fig, axes = plt.subplots(1, len(order), figsize=(5.2 * len(order), 4.2), squeeze=False)
    for ax, key in zip(axes.ravel(), order):
        im = _plot_heatmap(ax, spectra[key], wavelengths, t, key, cmap=SPECTRAL_CMAPS.get(key, "viridis"))
        ax.set_xlabel(xlabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        f"Native latent spectral components, sample {sample_index}",
        fontsize=13,
        fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = out_dir / f"sample_{sample_index:04d}_native_spectral_components.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _downsample_surface(image, wavelengths, time_values, max_wavelengths=120, max_time=120):
    wave_count, time_count = image.shape
    wave_idx = np.linspace(0, wave_count - 1, num=min(wave_count, max_wavelengths), dtype=int)
    time_idx = np.linspace(0, time_count - 1, num=min(time_count, max_time), dtype=int)
    z = image[np.ix_(wave_idx, time_idx)]
    x, y = np.meshgrid(time_values[time_idx], wavelengths[wave_idx])
    return x, y, z


def _save_native_spectral_domain_3d(data, out_dir, sample_index, sampling_rate, dpi):
    component_keys = _spectral_component_keys(data)
    spectra = {
        key: _load_spectrum(data, key, sample_index)
        for key in component_keys
    }
    spectra = {key: value for key, value in spectra.items() if value is not None}
    if not spectra:
        return None

    n_steps = next(iter(spectra.values())).shape[-1]
    wavelengths = (
        np.asarray(data["wavelengths"])
        if "wavelengths" in data
        else np.arange(next(iter(spectra.values())).shape[0]))
    t, xlabel = _time_axis(n_steps, sampling_rate)
    order = [key for key in component_keys if key in spectra]

    fig = plt.figure(figsize=(5.6 * len(order), 5.0))
    for idx, key in enumerate(order, start=1):
        ax = fig.add_subplot(1, len(order), idx, projection="3d")
        x, y, z = _downsample_surface(spectra[key], wavelengths, t)
        surface = ax.plot_surface(
            x,
            y,
            z,
            cmap=SPECTRAL_CMAPS.get(key, "viridis"),
            linewidth=0,
            antialiased=True,
            shade=True)
        ax.set_title(key, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel("Wavelength (nm)", labelpad=8)
        ax.set_zlabel("Intensity", labelpad=8)
        ax.view_init(elev=28, azim=-135)
        ax.tick_params(axis="both", labelsize=8)
        fig.colorbar(surface, ax=ax, shrink=0.55, pad=0.08)

    fig.suptitle(
        f"3D native latent spectral components, sample {sample_index}",
        fontsize=13,
        fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = out_dir / f"sample_{sample_index:04d}_native_spectral_components_3d.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _save_mean_curves(data, out_dir, sample_index, sampling_rate, dpi):
    del sampling_rate
    curves = {}
    for key in _spectral_component_keys(data):
        spectrum = _load_spectrum(data, key, sample_index)
        if spectrum is not None:
            curves[key] = spectrum.mean(axis=-1)
    if not curves:
        return None
    wavelengths = (
        np.asarray(data["wavelengths"])
        if "wavelengths" in data
        else np.arange(next(iter(curves.values())).shape[0]))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for key, value in curves.items():
        ax.plot(wavelengths, value, linewidth=2.0, label=key, color=SPECTRAL_COLORS.get(key))
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean spectral intensity")
    ax.set_title(f"Mean native spectra across time, sample {sample_index}", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = out_dir / f"sample_{sample_index:04d}_native_spectral_mean_curves.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize role-aware H decomposition in observation and spectral domains.")
    parser.add_argument("--npz", required=True, help="Path to exported spectral_outputs .npz file.")
    parser.add_argument("--out-dir", default=None, help="Output directory for PNG figures.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--channels", default="all", help="Comma-separated channel indices or 'all'.")
    parser.add_argument("--sampling-rate", type=float, default=None, help="Optional PPG sampling rate in Hz.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--include-3d", action="store_true", help="Also save 3D native spectral surface plots.")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir) if args.out_dir else npz_path.with_suffix("").parent / "h_decomposition_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=True) as data:
        written = []
        written.append(_save_observation_domain(data, out_dir, args.sample_index, args.channels, args.sampling_rate, args.dpi))
        projected = _save_projected_spectral_domain(data, out_dir, args.sample_index, args.sampling_rate, args.dpi)
        if projected is not None:
            written.append(projected)
        native = _save_native_spectral_domain(data, out_dir, args.sample_index, args.sampling_rate, args.dpi)
        if native is not None:
            written.append(native)
        if args.include_3d:
            native_3d = _save_native_spectral_domain_3d(data, out_dir, args.sample_index, args.sampling_rate, args.dpi)
            if native_3d is not None:
                written.append(native_3d)
        mean_curves = _save_mean_curves(data, out_dir, args.sample_index, args.sampling_rate, args.dpi)
        if mean_curves is not None:
            written.append(mean_curves)

    print("Saved figures:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
