"""
CP2PBR Blender Addon

This addon processes Cross-Polarized (CP) and Non-Cross-Polarized (NP) textures
to generate PBR maps like Metalness, Roughness, and Albedo.
"""

import bpy
import blf
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import StringProperty, PointerProperty, FloatProperty, BoolProperty, EnumProperty, FloatVectorProperty
from bpy_extras.io_utils import ImportHelper

import os
from array import array

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

_WAIT_CURSOR_DEPTH = 0
_WAIT_CURSOR_WINDOW = None
_PROGRESS_TOTAL = 1.0
_PROGRESS_VALUE = 0.0
_PROGRESS_LABEL = ""
_PROGRESS_OVERLAY_HANDLER = None
_PROGRESS_OVERLAY_AREA_PTR = None
_PROGRESS_OVERLAY_REGION_PTR = None
_PROGRESS_OVERLAY_X = 24
_PROGRESS_OVERLAY_Y = 24

# ----------------------------
# Helpers
# ----------------------------

def _ensure_dir(path):
    """Ensure the directory exists, creating it if necessary."""
    if path: os.makedirs(path, exist_ok=True)

def _norm_dir(path):
    """Normalize a Blender path to an absolute path."""
    return bpy.path.abspath(path) if path else ""

def _resolved_output_dir(settings):
    raw_output_dir = str(getattr(settings, "output_dir", "") or "").strip()
    raw_output_dir_l = raw_output_dir.lower()
    if (not raw_output_dir) or (raw_output_dir_l == "//set output folder"):
        return ""
    if "set output folder" in raw_output_dir_l:
        return ""
    return _norm_dir(raw_output_dir)

def _ext_lower(filepath):
    """Get the file extension in lowercase."""
    return os.path.splitext(filepath)[1].lower()

def _file_format_from_ext(ext):
    """Determine the Blender file format from the file extension."""
    if ext == ".png": return "PNG"
    if ext in {".jpg", ".jpeg"}: return "JPEG"
    if ext in {".tif", ".tiff"}: return "TIFF"
    if ext == ".exr": return "OPEN_EXR"
    if ext == ".tga": return "TARGA"
    if ext == ".bmp": return "BMP"
    return "PNG"

def _build_out_path(output_dir, stem, ext):
    """Build the output file path from directory, stem, and extension."""
    return os.path.join(output_dir, f"{stem}{ext}")

def _clamp01(x):
    """Clamp a value to the range [0.0, 1.0]."""
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _load_image(filepath, force_reload=False):
    """Load a Blender image from filepath, optionally forcing reload."""
    fp = bpy.path.abspath(filepath)
    if not os.path.isfile(fp): return None
    img = bpy.data.images.load(fp, check_existing=True)
    if force_reload:
        try:
            img.filepath = fp
            img.reload()
        except:
            pass
    _ = img.size[0]
    return img

def _new_image_like(name, w, h, *, float_buffer=False, alpha=True):
    """Create a new Blender image with specified name, size, and options."""
    img = bpy.data.images.new(name=name, width=w, height=h, alpha=alpha, float_buffer=float_buffer)
    return img

def _save_image_to(context, img, out_path, *, tiff_16bit=False, allow_resize=True):
    """Save a Blender image to the specified path, handling resize and format."""
    if context and hasattr(context, "scene"):
        s = getattr(context.scene, "cp2pbr_settings", None)
        if allow_resize and s and getattr(s, "enable_resize", False):
            tw = getattr(s, "target_width", 2048)
            cw, ch = img.size[0], img.size[1]
            if tw < cw:
                new_h = int((tw / cw) * ch)
                img.scale(tw, new_h)

        ext = _ext_lower(out_path)
        fmt = _file_format_from_ext(ext)
        img.file_format = fmt

        if context and hasattr(context, "scene"):
            scn = context.scene
            old_ff = scn.render.image_settings.file_format
            old_q = scn.render.image_settings.quality
            old_depth = scn.render.image_settings.color_depth
            
            try:
                scn.render.image_settings.file_format = fmt
                if fmt == "JPEG":
                    scn.render.image_settings.quality = 85
                if fmt == "TIFF" and tiff_16bit:
                    scn.render.image_settings.color_depth = '16'
                try:
                    img.save_render(out_path, scene=scn)
                except Exception:
                    img.filepath_raw = out_path
                    img.save()
            finally:
                scn.render.image_settings.file_format = old_ff
                scn.render.image_settings.quality = old_q
                scn.render.image_settings.color_depth = old_depth
        else:
            img.filepath_raw = out_path
            img.save()

def _normalized_rgb_weights(settings):
    weights = (
        max(0.0, float(settings.debug_weight_r)),
        max(0.0, float(settings.debug_weight_g)),
        max(0.0, float(settings.debug_weight_b)),
    )
    total = sum(weights)
    return tuple(weight / total for weight in weights) if total > 0.0 else (1.0 / 3.0,) * 3

def _weighted_luminance(r, g, b, weights):
    return weights[0] * r + weights[1] * g + weights[2] * b

def _srgb_to_linear(values):
    if _HAS_NUMPY and isinstance(values, np.ndarray):
        return np.where(
            values <= 0.04045,
            values / 12.92,
            ((values + 0.055) / 1.055) ** 2.4,
        )
    if values <= 0.04045:
        return values / 12.92
    return ((values + 0.055) / 1.055) ** 2.4

def _linear_to_srgb(values):
    if _HAS_NUMPY and isinstance(values, np.ndarray):
        return np.where(
            values <= 0.0031308,
            values * 12.92,
            1.055 * np.power(np.maximum(values, 0.0), 1.0 / 2.4) - 0.055,
        )
    if values <= 0.0031308:
        return values * 12.92
    return 1.055 * (values ** (1.0 / 2.4)) - 0.055

def _load_image_raw(filepath):
    """Load a private copy of an image with Non-Color so pixels hold the stored (encoded) values.

    The caller must release it with _discard_image.
    """
    fp = bpy.path.abspath(filepath) if filepath else ""
    if not fp or not os.path.isfile(fp):
        return None
    img = bpy.data.images.load(fp, check_existing=False)
    _set_colorspace(img, "Non-Color")
    _ = img.size[0]
    return img

def _discard_image(img):
    if img is None:
        return
    try:
        bpy.data.images.remove(img)
    except Exception:
        pass

_HIST_KINDS = ("np", "cp")
_HIST_LABELS = {"np": "NPC", "cp": "CP"}
_HIST_MAP_TYPES = {"NP_HIST": "np", "CP_HIST": "cp"}

def _hist_params(settings, kind):
    return (
        float(getattr(settings, f"{kind}_hist_black")),
        float(getattr(settings, f"{kind}_hist_white")),
        tuple(float(c) for c in getattr(settings, f"{kind}_hist_black_col")[:3]),
        tuple(float(c) for c in getattr(settings, f"{kind}_hist_white_col")[:3]),
    )

def _hist_modified(settings, kind):
    black, white, black_col, white_col = _hist_params(settings, kind)
    return (
        abs(black) > 1e-6
        or abs(white - 1.0) > 1e-6
        or any(abs(c) > 1e-6 for c in black_col)
        or any(abs(c - 1.0) > 1e-6 for c in white_col)
    )

def _adjusted_texture_path(settings, kind):
    out_dir = _norm_dir(settings.output_dir)
    if not out_dir:
        return ""
    name = "NP_Histogram_Adjusted" if kind == "np" else "CP_Histogram_Adjusted"
    return _build_out_path(os.path.join(out_dir, "debug maps"), name, ".tif")

def _source_texture_path(settings, kind):
    """Texture used by the debug/raw workflow: the adjusted one when the histogram was modified."""
    original = settings.np_path if kind == "np" else settings.cp_path
    if _hist_modified(settings, kind):
        adjusted = _adjusted_texture_path(settings, kind)
        if adjusted and os.path.isfile(adjusted):
            return adjusted
    return original

def _apply_hist_levels(rgb_encoded, settings, kind):
    """Stretch/compress the histogram in linear light, same math as the preview nodes."""
    black, white, black_col, white_col = _hist_params(settings, kind)
    span = max(white - black, 1e-4)
    black_col = np.array(black_col, dtype=np.float32)
    delta_col = np.array(white_col, dtype=np.float32) - black_col
    linear = _srgb_to_linear(np.clip(rgb_encoded, 0.0, 1.0))
    t = np.clip((linear - black) / span, 0.0, 1.0)
    out = np.clip(black_col + t * delta_col, 0.0, 1.0)
    return _linear_to_srgb(out).astype(np.float32)

def _debug_signature_path(settings):
    out_dir = _norm_dir(settings.output_dir)
    if not out_dir:
        return ""
    return os.path.join(out_dir, "debug maps", "Debug_Maps_Signature.txt")

def _debug_signature(settings):
    weights = _normalized_rgb_weights(settings)
    parts = [
        "weights=" + ",".join(f"{w:.6f}" for w in weights),
        f"cp_brightness={float(settings.cp_brightness_input):.6f}",
        f"rgb_diff={int(bool(settings.export_rgb_diff_maps))}",
        f"normalize={int(bool(settings.enable_normalization))}",
        "preset=" + (bpy.path.abspath(settings.histogram_params_file) if settings.reuse_histogram_params_from_file else ""),
    ]
    for kind in _HIST_KINDS:
        black, white, black_col, white_col = _hist_params(settings, kind)
        values = [black, white, *black_col, *white_col]
        parts.append(f"{kind}_hist=" + ",".join(f"{v:.6f}" for v in values))
    return "\n".join(parts) + "\n"

def _write_debug_signature(settings):
    path = _debug_signature_path(settings)
    if not path:
        return
    try:
        _ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            f.write(_debug_signature(settings))
    except Exception:
        pass

def _debug_signature_matches(settings):
    path = _debug_signature_path(settings)
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read() == _debug_signature(settings)
    except Exception:
        return False

def _pixels_count(img):
    """Get the total number of pixel components (width * height * 4)."""
    w, h = img.size[0], img.size[1]
    return w * h * 4

def _read_pixels_fast_fallback(img):
    """Read image pixels into a buffer, with fallback for older Blender versions."""
    n = _pixels_count(img)
    buf = array('f', [0.0]) * n
    px = img.pixels
    if hasattr(px, "foreach_get"): px.foreach_get(buf)
    else: buf[:] = array('f', px[:])
    return buf

def _write_pixels_fast(img, buf):
    """Write pixel buffer back to image, with fallback."""
    px = img.pixels
    if hasattr(px, "foreach_set"): px.foreach_set(buf)
    else: img.pixels = buf.tolist() if _HAS_NUMPY else buf.tolist()
    img.update()

def _safe_factor(adj):
    """Compute a safe adjustment factor, clamped to non-negative."""
    f = 1.0 + float(adj)
    return 0.0 if f < 0.0 else f

def _best_ext(settings):
    """Get the best file extension based on export format settings."""
    fmt = getattr(settings, "export_format", "PNG")
    if fmt == 'JPEG': return '.jpg'
    if fmt == 'TIFF': return '.tif'
    return '.png'

def _apply_resize_limit(settings, width, height):
    width = max(1, int(width))
    height = max(1, int(height))
    if getattr(settings, "enable_resize", False):
        target_width = max(1, int(getattr(settings, "target_width", width)))
        if width > target_width:
            scale = target_width / float(width)
            width = target_width
            height = max(1, int(round(height * scale)))
    return width, height

def _export_image_copy(context, source_image, out_path):
    temp_img = source_image.copy()
    try:
        _save_image_to(context, temp_img, out_path)
    finally:
        try:
            bpy.data.images.remove(temp_img)
        except Exception:
            pass

def _np_light_path(settings):
    out_dir = _norm_dir(settings.output_dir)
    if not out_dir or not settings.np_path: return ""
    d = os.path.join(out_dir, "debug maps")
    _ensure_dir(d)
    return _build_out_path(d, "NP_Light", ".tif")

def _cp_light_path(settings):
    out_dir = _norm_dir(settings.output_dir)
    if not out_dir or not settings.cp_path: return ""
    d = os.path.join(out_dir, "debug maps")
    _ensure_dir(d)
    return _build_out_path(d, "CP_Light", ".tif")

def _guess_map_path(settings, stem):
    out_dir = _norm_dir(settings.output_dir)
    if not out_dir: return ""
    d = os.path.join(out_dir, "debug maps")
    candidates = [".tif", ".tiff", _best_ext(settings), ".png", ".jpg", ".jpeg", ".exr", ".tga", ".bmp"]
    seen = set()
    for ext in candidates:
        ext = ext.lower()
        if ext in seen:
            continue
        seen.add(ext)
        path = _build_out_path(d, stem, ext)
        if os.path.isfile(path):
            return path
    return ""

def _processed_dir(settings):
    out_dir = _norm_dir(settings.output_dir)
    if not out_dir: return ""
    return os.path.join(out_dir, "processed textures")

def _set_colorspace(img, name):
    try: img.colorspace_settings.name = name
    except Exception: pass

_CP_TEXTURE_NODE_NAME = "CP2PBR_CP_Texture"
_FAKE_NORMAL_NODE_NAME = "CP2PBR_FakeNormalBump"
_CUSTOM_NORMAL_TEXTURE_NODE_NAME = "CP2PBR_CustomNormalTexture"
_CUSTOM_NORMAL_MAP_NODE_NAME = "CP2PBR_CustomNormalMap"
_NORMAL_LINK_BACKUP_PROP = "_cp2pbr_normal_link_backup"
_BAKED_NODE_NAMES = {
    "Base Color": "CP2PBR_Baked_BaseColor",
    "Roughness": "CP2PBR_Baked_Roughness",
    "Metallic": "CP2PBR_Baked_Metalness",
    "NormalTexture": "CP2PBR_Baked_NormalTexture",
    "NormalMap": "CP2PBR_Baked_NormalMap",
}

def _debug_output_targets(settings):
    out_dir = _norm_dir(settings.output_dir)
    if not out_dir:
        return []
    debug_dir = os.path.join(out_dir, "debug maps")
    ext = ".tif"
    targets = [
        _build_out_path(debug_dir, "NP_Light", ext),
        _build_out_path(debug_dir, "CP_Light", ext),
        _build_out_path(debug_dir, "Debug_Subtr_Normalized", ext),
        _build_out_path(debug_dir, "Debug_Subtr_Normalized_R", ext),
        _build_out_path(debug_dir, "Debug_Subtr_Normalized_G", ext),
        _build_out_path(debug_dir, "Debug_Subtr_Normalized_B", ext),
    ]
    for kind in _HIST_KINDS:
        if _hist_modified(settings, kind):
            targets.append(_adjusted_texture_path(settings, kind))
    if settings.export_rgb_diff_maps:
        targets.extend(_build_out_path(debug_dir, _rgb_diff_stem(n), ext) for n in _RGB_DIFF_NAMES)
    return targets

def _debug_maps_complete(settings):
    targets = _debug_output_targets(settings)
    return bool(targets) and all(os.path.isfile(path) for path in targets)

def _histogram_params_path(settings):
    out_dir = _resolved_output_dir(settings)
    if not out_dir:
        return ""
    debug_dir = os.path.join(out_dir, "debug maps")
    return os.path.join(debug_dir, "Histogram_Stretch_Compression_Params.txt")

def _calibration_params_path(settings):
    out_dir = _resolved_output_dir(settings)
    if not out_dir:
        return ""
    debug_dir = os.path.join(out_dir, "debug maps")
    return os.path.join(debug_dir, "Shader_Calibration_Coefficients.txt")

def _write_histogram_params_file(settings, debug_diff_min, debug_diff_max):
    out_path = _histogram_params_path(settings)
    if not out_path:
        return (False, "Please set an output folder first.")
    try:
        _ensure_dir(os.path.dirname(out_path))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# CP2PBR debug map histogram stretch/compression parameters\n")
            f.write(f"debug_diff_min={float(debug_diff_min):.6f}\n")
            f.write(f"debug_diff_max={float(debug_diff_max):.6f}\n")
    except Exception as e:
        return (False, f"Could not write histogram parameters file: {str(e)}")
    return (True, out_path)

def _load_histogram_params_file(filepath):
    if not filepath:
        return (None, "Please select a histogram parameters text file.")
    full_path = bpy.path.abspath(filepath)
    if not os.path.isfile(full_path):
        return (None, "Histogram parameters file not found.")

    wanted_keys = {"debug_diff_min", "debug_diff_max"}
    parsed = {}
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, raw_value = line.split("=", 1)
                key = key.strip()
                if key in wanted_keys:
                    parsed[key] = float(raw_value.strip())
    except Exception as e:
        return (None, f"Could not read histogram parameters file: {str(e)}")

    missing = [key for key in wanted_keys if key not in parsed]
    if missing:
        return (None, "Histogram parameters file is incomplete.")
    return (parsed, full_path)

def _collect_calibration_coeffs(settings):
    return {
        "metalness_black": float(settings.metalness_black),
        "metalness_grey": float(settings.metalness_grey),
        "metalness_white": float(settings.metalness_white),
        "metalness_black_col": tuple(float(v) for v in settings.metalness_black_col),
        "metalness_white_col": tuple(float(v) for v in settings.metalness_white_col),
        "roughness_black": float(settings.roughness_black),
        "roughness_grey": float(settings.roughness_grey),
        "roughness_white": float(settings.roughness_white),
        "roughness_black_col": tuple(float(v) for v in settings.roughness_black_col),
        "roughness_white_col": tuple(float(v) for v in settings.roughness_white_col),
        "metal_hsv_hue": float(settings.metal_hsv_hue),
        "metal_hsv_saturation": float(settings.metal_hsv_saturation),
        "metal_hsv_value": float(settings.metal_hsv_value),
    }

def _write_calibration_coeffs_file(settings):
    raw_output_dir = str(getattr(settings, "output_dir", "") or "").strip()
    if (not raw_output_dir) or (raw_output_dir.lower() == "//set output folder"):
        return (False, "no output folder is found")
    out_path = _calibration_params_path(settings)
    if not out_path:
        return (False, "no output folder is found")
    coeffs = _collect_calibration_coeffs(settings)
    try:
        _ensure_dir(os.path.dirname(out_path))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# CP2PBR calibration coefficients\n")
            for key, value in coeffs.items():
                if isinstance(value, tuple):
                    f.write(f"{key}={','.join(f'{v:.6f}' for v in value)}\n")
                else:
                    f.write(f"{key}={value:.6f}\n")
    except PermissionError:
        return (False, "no output folder is found")
    except Exception as e:
        return (False, f"Could not write calibration file: {str(e)}")
    return (True, out_path)

def _parse_vec4(raw_value):
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 4:
        raise ValueError("Expected 4 comma-separated values")
    return tuple(float(part) for part in parts)

def _load_calibration_coeffs_file(filepath):
    if not filepath:
        return (None, "Please select a calibration text file.")
    full_path = bpy.path.abspath(filepath)
    if not os.path.isfile(full_path):
        return (None, "Calibration file not found.")

    scalar_keys = {
        "metalness_black",
        "metalness_grey",
        "metalness_white",
        "roughness_black",
        "roughness_grey",
        "roughness_white",
        "metal_hsv_hue",
        "metal_hsv_saturation",
        "metal_hsv_value",
    }
    vec_keys = {"metalness_black_col", "metalness_white_col", "roughness_black_col", "roughness_white_col"}
    parsed = {}

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, raw_value = line.split("=", 1)
                key = key.strip()
                raw_value = raw_value.strip()
                if key in scalar_keys:
                    parsed[key] = float(raw_value)
                elif key in vec_keys:
                    parsed[key] = _parse_vec4(raw_value)
    except Exception as e:
        return (None, f"Could not read calibration file: {str(e)}")

    missing = [key for key in (scalar_keys | vec_keys) if key not in parsed]
    if missing:
        return (None, "Calibration file is incomplete.")
    return (parsed, full_path)

def _apply_calibration_coeffs(settings, coeffs):
    settings.metalness_black = coeffs["metalness_black"]
    settings.metalness_grey = coeffs["metalness_grey"]
    settings.metalness_white = coeffs["metalness_white"]
    settings.metalness_black_col = coeffs["metalness_black_col"]
    settings.metalness_white_col = coeffs["metalness_white_col"]
    settings.roughness_black = coeffs["roughness_black"]
    settings.roughness_grey = coeffs["roughness_grey"]
    settings.roughness_white = coeffs["roughness_white"]
    settings.roughness_black_col = coeffs["roughness_black_col"]
    settings.roughness_white_col = coeffs["roughness_white_col"]
    settings.metal_hsv_hue = coeffs["metal_hsv_hue"]
    settings.metal_hsv_saturation = coeffs["metal_hsv_saturation"]
    settings.metal_hsv_value = coeffs["metal_hsv_value"]

_RAW_PANEL_BOOL_KEYS = (
    "reuse_existing_debug_maps", "enable_normalization", "reuse_histogram_params_from_file",
    "export_rgb_diff_maps",
)
_RAW_PANEL_FLOAT_KEYS = ("debug_weight_r", "debug_weight_g", "debug_weight_b") + tuple(
    f"{k}_hist_{n}" for k in _HIST_KINDS for n in ("black", "white")
)
_RAW_PANEL_VEC_KEYS = tuple(f"{k}_hist_{n}_col" for k in _HIST_KINDS for n in ("black", "white"))

def _hist_preset_path(settings):
    out_dir = _resolved_output_dir(settings)
    if not out_dir:
        return ""
    return os.path.join(out_dir, "debug maps", "Raw_Debug_Settings.txt")

def _write_hist_preset_file(settings):
    raw_output_dir = str(getattr(settings, "output_dir", "") or "").strip()
    out_path = _hist_preset_path(settings)
    if not out_path or raw_output_dir.lower() == "//set output folder":
        return (False, "no output folder is found")
    try:
        _ensure_dir(os.path.dirname(out_path))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# CP2PBR raw shader + debug maps settings\n")
            for key in _RAW_PANEL_BOOL_KEYS:
                f.write(f"{key}={int(bool(getattr(settings, key)))}\n")
            for key in _RAW_PANEL_FLOAT_KEYS:
                f.write(f"{key}={float(getattr(settings, key)):.6f}\n")
            for key in _RAW_PANEL_VEC_KEYS:
                f.write(f"{key}={','.join(f'{float(v):.6f}' for v in getattr(settings, key))}\n")
            f.write(f"histogram_params_file={settings.histogram_params_file}\n")
    except Exception as e:
        return (False, f"Could not write settings file: {str(e)}")
    return (True, out_path)

def _load_hist_preset_file(filepath):
    if not filepath:
        return (None, "Please select a settings text file.")
    full_path = bpy.path.abspath(filepath)
    if not os.path.isfile(full_path):
        return (None, "Settings file not found.")
    parsed = {}
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, raw_value = (p.strip() for p in line.split("=", 1))
                if key in _RAW_PANEL_BOOL_KEYS:
                    parsed[key] = raw_value.lower() in {"1", "true", "yes"}
                elif key in _RAW_PANEL_FLOAT_KEYS:
                    parsed[key] = float(raw_value)
                elif key in _RAW_PANEL_VEC_KEYS:
                    parsed[key] = _parse_vec4(raw_value)
                elif key == "histogram_params_file":
                    parsed[key] = raw_value
    except Exception as e:
        return (None, f"Could not read settings file: {str(e)}")
    if not parsed:
        return (None, "Settings file contains no valid values.")
    return (parsed, full_path)

def _apply_hist_preset(settings, coeffs):
    for key, value in coeffs.items():
        setattr(settings, key, value)

def _remove_small_clusters(values, min_size, level=0.02):
    """Set to 0 the 4-connected clusters of pixels above `level` that have fewer than `min_size` pixels."""
    mask = values > level
    try:
        from scipy import ndimage
        labels, count = ndimage.label(mask)
        if count == 0:
            return values
        sizes = np.bincount(labels.ravel())
        small = sizes < min_size
        small[0] = False
        out = values.copy()
        out[small[labels]] = 0.0
        return out
    except ImportError:
        pass

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out = values.copy()
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        cluster = [(y0, x0)]
        stack = [(y0, x0)]
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
                    cluster.append((ny, nx))
        if len(cluster) < min_size:
            for y, x in cluster:
                out[y, x] = 0.0
    return out

def _processed_output_targets(settings, suffixes):
    processed_dir = _processed_dir(settings)
    if not processed_dir:
        return []
    ext = _best_ext(settings)
    return [_build_out_path(processed_dir, suffix, ext) for suffix in suffixes]

def _existing_output_paths(paths):
    return [path for path in paths if path and os.path.isfile(path)]

def _remove_node_and_links(node_tree, node):
    links = node_tree.links
    for sock in node.inputs:
        for link in list(sock.links):
            links.remove(link)
    for sock in node.outputs:
        for link in list(sock.links):
            links.remove(link)
    node_tree.nodes.remove(node)

def _unlink_socket_links(links, socket):
    if not socket:
        return
    for link in list(socket.links):
        links.remove(link)

def _clear_bsdf_input(nodes, links, bsdf, socket_name):
    socket = bsdf.inputs.get(socket_name)
    if not socket:
        return
    for link in list(socket.links):
        node = link.from_node
        links.remove(link)
        if node.type in {'TEX_IMAGE', 'NORMAL_MAP', 'BUMP'} and not any(out.is_linked for out in node.outputs):
            try:
                nodes.remove(node)
            except Exception:
                pass

def _expected_baked_map_paths(settings, suffixes):
    processed_dir = _processed_dir(settings)
    if not processed_dir:
        return {}
    ext = _best_ext(settings)
    return {suffix: _build_out_path(processed_dir, suffix, ext) for suffix in suffixes}

def _apply_baked_maps_to_material(material, baked_images_paths):
    if not material:
        return False

    material.use_nodes = True

    loaded_images = {}
    for suffix, img_path in baked_images_paths.items():
        img = _load_image(img_path)
        if not img:
            return False
        loaded_images[suffix] = img

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    for node in list(nodes):
        _remove_node_and_links(material.node_tree, node)

    out_node = nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (350, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    links.new(bsdf.outputs[0], out_node.inputs[0])

    def connect_baked_map(sock_name, image, loc_y, node_name, is_noncolor=True):
        sock = bsdf.inputs.get(sock_name)
        if not sock:
            return

        t_node = nodes.new("ShaderNodeTexImage")
        t_node.name = t_node.label = node_name
        t_node.location = (bsdf.location.x - 300, bsdf.location.y + loc_y)
        t_node.image = image
        _set_colorspace(t_node.image, "Non-Color" if is_noncolor else "sRGB")

        _unlink_socket_links(links, sock)
        links.new(t_node.outputs["Color"], sock)

    if 'BaseColor' in loaded_images:
        connect_baked_map("Base Color", loaded_images['BaseColor'], 380, _BAKED_NODE_NAMES["Base Color"], False)

    if 'Metalness' in loaded_images:
        connect_baked_map("Metallic", loaded_images['Metalness'], 0, _BAKED_NODE_NAMES["Metallic"], True)

    if 'Roughness' in loaded_images:
        connect_baked_map("Roughness", loaded_images['Roughness'], -380, _BAKED_NODE_NAMES["Roughness"], True)

    if 'Normal' in loaded_images:
        sock = bsdf.inputs.get("Normal")
        if sock:
            t_n = nodes.new("ShaderNodeTexImage")
            t_n.name = t_n.label = _BAKED_NODE_NAMES["NormalTexture"]
            t_n.location = (bsdf.location.x - 600,             bsdf.location.y - 760)
            t_n.image = loaded_images['Normal']
            _set_colorspace(t_n.image, "Non-Color")

            n_map = nodes.new("ShaderNodeNormalMap")
            n_map.name = n_map.label = _BAKED_NODE_NAMES["NormalMap"]
            n_map.location = (bsdf.location.x - 300,             bsdf.location.y - 760)

            _unlink_socket_links(links, sock)
            links.new(t_n.outputs["Color"], n_map.inputs["Color"])
            links.new(n_map.outputs["Normal"], sock)

    return True

def _get_direct_normal_image_node(socket):
    if not socket or not socket.is_linked:
        return None
    node = socket.links[0].from_node
    if node.type == 'NORMAL_MAP' and node.inputs['Color'].is_linked:
        tex_node = node.inputs['Color'].links[0].from_node
        if tex_node.type == 'TEX_IMAGE' and tex_node.image:
            return tex_node
    return None

def _linked_bake_suffixes(material):
    if not material or not material.use_nodes:
        return []
    nodes = material.node_tree.nodes
    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not bsdf:
        return []

    suffixes = []
    for sock_name, suffix in (
        ("Base Color", "BaseColor"),
        ("Metallic", "Metalness"),
        ("Roughness", "Roughness"),
        ("Normal", "Normal"),
    ):
        socket = bsdf.inputs.get(sock_name)
        if socket and socket.is_linked:
            suffixes.append(suffix)
    return suffixes

def _configure_fake_normal(node_tree, bsdf, cp_color_socket, enabled, distance=0.002):
    normal_input = bsdf.inputs.get("Normal")
    if not normal_input:
        return

    nodes = node_tree.nodes
    links = node_tree.links
    bump = nodes.get(_FAKE_NORMAL_NODE_NAME)
    material = getattr(node_tree, "id_data", None)

    def _backup_existing_normal_link():
        if not material:
            return
        if not normal_input.is_linked:
            return
        normal_link = normal_input.links[0]
        if normal_link.from_node == bump:
            return
        material[_NORMAL_LINK_BACKUP_PROP] = f"{normal_link.from_node.name}|{normal_link.from_socket.name}"
        for link in list(normal_input.links):
            links.remove(link)

    def _restore_backup_normal_link():
        if not material:
            return
        if normal_input.is_linked:
            return
        backup_value = material.get(_NORMAL_LINK_BACKUP_PROP)
        if not backup_value:
            return

        try:
            node_name, socket_name = str(backup_value).split("|", 1)
        except ValueError:
            material.pop(_NORMAL_LINK_BACKUP_PROP, None)
            return

        source_node = nodes.get(node_name)
        if not source_node:
            material.pop(_NORMAL_LINK_BACKUP_PROP, None)
            return

        source_socket = source_node.outputs.get(socket_name)
        if not source_socket:
            material.pop(_NORMAL_LINK_BACKUP_PROP, None)
            return

        links.new(source_socket, normal_input)
        material.pop(_NORMAL_LINK_BACKUP_PROP, None)

    if not enabled:
        if bump:
            _remove_node_and_links(node_tree, bump)
        _restore_backup_normal_link()
        return

    if cp_color_socket is None:
        return

    if not bump:
        bump = nodes.new("ShaderNodeBump")
        bump.name = bump.label = _FAKE_NORMAL_NODE_NAME
        bump.location = (bsdf.location.x - 320, bsdf.location.y - 570)

    bump.inputs["Distance"].default_value = distance
    bump.inputs["Strength"].default_value = 1.0

    for link in list(bump.inputs["Height"].links):
        links.remove(link)
    links.new(cp_color_socket, bump.inputs["Height"])

    _backup_existing_normal_link()
    if normal_input.is_linked:
        for link in list(normal_input.links):
            links.remove(link)
    links.new(bump.outputs["Normal"], normal_input)

def _show_popup_message(context, title, message, icon='INFO'):
    wm = getattr(context, "window_manager", None)
    if not wm:
        return

    def draw(self, _context):
        self.layout.label(text=message)

    wm.popup_menu(draw, title=title, icon=icon)

def _mesh_object_poll(self, obj):
    return bool(obj and obj.type == 'MESH')

def _get_input_mesh(context):
    scene = getattr(context, "scene", None)
    settings = getattr(scene, "cp2pbr_settings", None)
    obj = getattr(settings, "input_mesh", None)
    if not obj or obj.type != 'MESH':
        return None
    if scene and scene.objects.get(obj.name) is None:
        return None
    return obj

def _require_input_mesh(context, operator=None, action="running this command"):
    obj = _get_input_mesh(context)
    if obj:
        return obj

    message = f"Set Input Mesh in the Inputs panel before {action}."
    _show_popup_message(context, "CP2PBR Alert", message, icon='ERROR')
    if operator:
        operator.report({'ERROR'}, message)
    return None

def _sync_active_selected_object(context, obj):
    if not obj:
        return
    view_layer = getattr(context, "view_layer", None)
    if view_layer:
        try:
            view_layer.objects.active = obj
        except Exception:
            pass
    if not obj.select_get():
        try:
            obj.select_set(True)
        except Exception:
            pass

def _redraw_progress_ui():
    try:
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    except Exception:
        pass

def _draw_progress_overlay():
    area = getattr(bpy.context, "area", None)
    region = getattr(bpy.context, "region", None)
    if not area or not region:
        return
    if area.as_pointer() != _PROGRESS_OVERLAY_AREA_PTR or region.as_pointer() != _PROGRESS_OVERLAY_REGION_PTR:
        return

    font_id = 0
    percent = 0
    if _PROGRESS_TOTAL > 0.0:
        percent = int(round((_PROGRESS_VALUE / _PROGRESS_TOTAL) * 100.0))
    percent = max(0, min(100, percent))
    percent_text = f"{percent}%"
    label_text = (_PROGRESS_LABEL or "Working...").strip()

    try:
        blf.size(font_id, 16)
    except TypeError:
        blf.size(font_id, 16, 72)
    label_w, label_h = blf.dimensions(font_id, label_text)

    try:
        blf.size(font_id, 30)
    except TypeError:
        blf.size(font_id, 30, 72)
    percent_w, percent_h = blf.dimensions(font_id, percent_text)

    percent_x = int((region.width - percent_w) * 0.5)
    percent_y = int((region.height - percent_h) * 0.5)
    label_x = int((region.width - label_w) * 0.5)
    label_y = percent_y + int(percent_h) + 12

    def draw_text(text, x, y, size, color):
        try:
            blf.size(font_id, size)
        except TypeError:
            blf.size(font_id, size, 72)
        blf.position(font_id, x + 1, y - 1, 0)
        blf.color(font_id, 0.0, 0.0, 0.0, 0.92)
        blf.draw(font_id, text)
        blf.position(font_id, x, y, 0)
        blf.color(font_id, *color)
        blf.draw(font_id, text)

    draw_text(label_text, label_x, label_y, 16, (0.88, 0.93, 1.0, 1.0))
    draw_text(percent_text, percent_x, percent_y, 30, (1.0, 1.0, 1.0, 1.0))

def _extract_light_image_to_file(context, src_img, out_path, out_ext, weights):
    w, h = src_img.size[0], src_img.size[1]
    want_float = True
    dst = _new_image_like(os.path.splitext(os.path.basename(out_path))[0], w, h, float_buffer=want_float, alpha=True)
    _set_colorspace(dst, "Non-Color")

    if _HAS_NUMPY:
        buf = np.empty(w * h * 4, dtype=np.float32)
        px = src_img.pixels
        if hasattr(px, "foreach_get"):
            px.foreach_get(buf)
        else:
            buf[:] = np.array(px[:], dtype=np.float32)

        red = np.clip(buf[0::4], 0.0, 1.0)
        green = np.clip(buf[1::4], 0.0, 1.0)
        blue = np.clip(buf[2::4], 0.0, 1.0)
        alpha = buf[3::4]
        del buf

        luminance = np.clip(
            red * weights[0] + green * weights[1] + blue * weights[2],
            0.0,
            1.0,
        )
        del red, green, blue

        out_buf = np.empty(w * h * 4, dtype=np.float32)
        out_buf[0::4] = luminance
        out_buf[1::4] = luminance
        out_buf[2::4] = luminance
        out_buf[3::4] = alpha
        del luminance, alpha
    else:
        src_buf = _read_pixels_fast_fallback(src_img)
        out_buf = array('f', [0.0]) * (w * h * 4)
        for i in range(0, len(src_buf), 4):
            y_val = _clamp01(_weighted_luminance(
                _clamp01(src_buf[i]),
                _clamp01(src_buf[i+1]),
                _clamp01(src_buf[i+2]),
                weights,
            ))
            out_buf[i:i+4] = array('f', [y_val, y_val, y_val, _clamp01(src_buf[i+3])])

    _write_pixels_fast(dst, out_buf)
    try:
        _save_image_to(context, dst, out_path, tiff_16bit=True)
    finally:
        _discard_image(dst)
    return out_path

def _export_adjusted_textures(context, s):
    """Write the histogram-adjusted NPC/CP textures (full resolution, 16-bit TIFF) when modified."""
    for kind in _HIST_KINDS:
        if not _hist_modified(s, kind):
            continue
        out_path = _adjusted_texture_path(s, kind)
        if not out_path:
            return (False, "Please set an output folder first.")
        _ensure_dir(os.path.dirname(out_path))
        src_path = s.np_path if kind == "np" else s.cp_path
        src = _load_image_raw(src_path)
        if src is None:
            return (False, f"Could not load the {_HIST_LABELS[kind]} texture.")
        dst = None
        try:
            w, h = src.size[0], src.size[1]
            count = w * h * 4
            dst = _new_image_like(os.path.splitext(os.path.basename(out_path))[0], w, h, float_buffer=True, alpha=True)
            _set_colorspace(dst, "Non-Color")
            if _HAS_NUMPY:
                buf = np.empty(count, dtype=np.float32)
                if hasattr(src.pixels, "foreach_get"):
                    src.pixels.foreach_get(buf)
                else:
                    buf[:] = np.array(src.pixels[:], dtype=np.float32)
                px = buf.reshape(-1, 4)
                px[:, :3] = _apply_hist_levels(px[:, :3], s, kind)
                _write_pixels_fast(dst, buf)
            else:
                black, white, black_col, white_col = _hist_params(s, kind)
                span = max(white - black, 1e-4)
                buf = _read_pixels_fast_fallback(src)
                for i in range(0, count, 4):
                    for c in range(3):
                        lin = _srgb_to_linear(_clamp01(buf[i + c]))
                        t = _clamp01((lin - black) / span)
                        value = _clamp01(black_col[c] + t * (white_col[c] - black_col[c]))
                        buf[i + c] = _linear_to_srgb(value)
                _write_pixels_fast(dst, buf)
            _save_image_to(context, dst, out_path, tiff_16bit=True, allow_resize=False)
        finally:
            _discard_image(src)
            _discard_image(dst)
    return (True, "")

def _export_rgb_debug_channels(context, s, brightness_adj):
    np_img = _load_image_raw(_source_texture_path(s, "np"))
    cp_img = _load_image_raw(_source_texture_path(s, "cp"))
    try:
        return _export_rgb_debug_channels_impl(context, s, brightness_adj, np_img, cp_img)
    finally:
        _discard_image(np_img)
        _discard_image(cp_img)

def _export_rgb_debug_channels_impl(context, s, brightness_adj, np_img, cp_img):
    if np_img is None or cp_img is None:
        return (False, "Could not load source textures for RGB debug maps.")

    w, h = np_img.size[0], np_img.size[1]
    if cp_img.size[0] != w or cp_img.size[1] != h:
        return (False, "Input textures must have matching dimensions to export RGB debug maps.")

    factor = _safe_factor(brightness_adj)
    ext = ".tif"
    if _HAS_NUMPY:
        count = w * h * 4
        np_buf = np.empty(count, dtype=np.float32)
        cp_buf = np.empty(count, dtype=np.float32)
        if hasattr(np_img.pixels, "foreach_get"):
            np_img.pixels.foreach_get(np_buf)
        else:
            np_buf[:] = np.array(np_img.pixels[:], dtype=np.float32)
        if hasattr(cp_img.pixels, "foreach_get"):
            cp_img.pixels.foreach_get(cp_buf)
        else:
            cp_buf[:] = np.array(cp_img.pixels[:], dtype=np.float32)

        for channel, name in enumerate(("R", "G", "B")):
            base = _srgb_to_linear(np.clip(np_buf[channel::4], 0.0, 1.0))
            cp_channel = _srgb_to_linear(np.clip(cp_buf[channel::4], 0.0, 1.0))
            blend = np.clip(cp_channel * factor, 0.0, 1.0)
            diff = base - blend
            if s.enable_normalization:
                d_min, d_max = float(diff.min()), float(diff.max())
            else:
                d_min, d_max = 0.0, 1.0
            if d_max > d_min:
                diff = (diff - d_min) / (d_max - d_min)
            else:
                diff = np.zeros_like(diff)

            out_buf = np.empty(count, dtype=np.float32)
            out_buf[0::4] = diff
            out_buf[1::4] = diff
            out_buf[2::4] = diff
            out_buf[3::4] = 1.0
            dst = _new_image_like(f"Debug_Subtr_Normalized_{name}", w, h, float_buffer=True, alpha=True)
            _set_colorspace(dst, "Non-Color")
            _write_pixels_fast(dst, out_buf)
            out_path = _build_out_path(
                os.path.join(_norm_dir(s.output_dir), "debug maps"),
                f"Debug_Subtr_Normalized_{name}",
                ext,
            )
            try:
                _save_image_to(context, dst, out_path, tiff_16bit=True)
            finally:
                bpy.data.images.remove(dst)
    else:
        np_buf = _read_pixels_fast_fallback(np_img)
        cp_buf = _read_pixels_fast_fallback(cp_img)
        for channel, name in enumerate(("R", "G", "B")):
            values = []
            for i in range(channel, len(np_buf), 4):
                values.append(
                    _srgb_to_linear(_clamp01(np_buf[i])) -
                    _clamp01(_srgb_to_linear(_clamp01(cp_buf[i])) * factor)
                )
            d_min, d_max = min(values), max(values)
            if not s.enable_normalization:
                d_min, d_max = 0.0, 1.0
            out_buf = array('f', [0.0]) * (w * h * 4)
            for pixel, value in enumerate(values):
                normalized = (value - d_min) / (d_max - d_min) if d_max > d_min else 0.0
                idx = pixel * 4
                out_buf[idx:idx+4] = array('f', [normalized, normalized, normalized, 1.0])
            dst = _new_image_like(f"Debug_Subtr_Normalized_{name}", w, h, float_buffer=True, alpha=True)
            _set_colorspace(dst, "Non-Color")
            _write_pixels_fast(dst, out_buf)
            out_path = _build_out_path(
                os.path.join(_norm_dir(s.output_dir), "debug maps"),
                f"Debug_Subtr_Normalized_{name}",
                ext,
            )
            try:
                _save_image_to(context, dst, out_path, tiff_16bit=True)
            finally:
                bpy.data.images.remove(dst)
    return (True, "")

_RGB_DIFF_NAMES = ("R", "G", "B", "Sum")

def _rgb_diff_stem(name):
    return f"Debug_RGB_Diff_{name}"

def _export_rgb_difference_maps(context, s):
    """Greyscale |NPC - CP| per RGB channel (linear light) plus their sum, as 16-bit TIFF."""
    if not _HAS_NUMPY:
        return (False, "NumPy is required for the RGB difference maps.")
    np_img = _load_image_raw(_source_texture_path(s, "np"))
    cp_img = _load_image_raw(_source_texture_path(s, "cp"))
    try:
        if np_img is None or cp_img is None:
            return (False, "Could not load source textures for RGB difference maps.")
        w, h = np_img.size[0], np_img.size[1]
        if cp_img.size[0] != w or cp_img.size[1] != h:
            return (False, "Input textures must have matching dimensions to export RGB difference maps.")
        count = w * h * 4
        np_buf = np.empty(count, dtype=np.float32)
        cp_buf = np.empty(count, dtype=np.float32)
        np_img.pixels.foreach_get(np_buf)
        cp_img.pixels.foreach_get(cp_buf)

        diffs = []
        for channel in range(3):
            a = _srgb_to_linear(np.clip(np_buf[channel::4], 0.0, 1.0))
            b = _srgb_to_linear(np.clip(cp_buf[channel::4], 0.0, 1.0))
            diffs.append(np.abs(a - b))
        diffs.append(diffs[0] + diffs[1] + diffs[2])

        debug_dir = os.path.join(_norm_dir(s.output_dir), "debug maps")
        _ensure_dir(debug_dir)
        for name, diff in zip(_RGB_DIFF_NAMES, diffs):
            d_max = float(diff.max()) if s.enable_normalization else 1.0
            norm = diff / d_max if d_max > 0.0 else np.zeros_like(diff)
            out_buf = np.empty(count, dtype=np.float32)
            out_buf[0::4] = norm
            out_buf[1::4] = norm
            out_buf[2::4] = norm
            out_buf[3::4] = 1.0
            dst = _new_image_like(_rgb_diff_stem(name), w, h, float_buffer=True, alpha=True)
            try:
                _set_colorspace(dst, "Non-Color")
                _write_pixels_fast(dst, out_buf)
                _save_image_to(context, dst, _build_out_path(debug_dir, _rgb_diff_stem(name), ".tif"), tiff_16bit=True)
            finally:
                _discard_image(dst)
    finally:
        _discard_image(np_img)
        _discard_image(cp_img)
    return (True, "")

def _ensure_progress_overlay_handler():
    global _PROGRESS_OVERLAY_HANDLER

    if _PROGRESS_OVERLAY_HANDLER is None:
        _PROGRESS_OVERLAY_HANDLER = bpy.types.SpaceView3D.draw_handler_add(
            _draw_progress_overlay,
            (),
            'WINDOW',
            'POST_PIXEL',
        )

def _remove_progress_overlay_handler():
    global _PROGRESS_OVERLAY_HANDLER

    if _PROGRESS_OVERLAY_HANDLER is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_PROGRESS_OVERLAY_HANDLER, 'WINDOW')
        except Exception:
            pass
        _PROGRESS_OVERLAY_HANDLER = None

def _remember_progress_anchor(context, event=None):
    global _PROGRESS_OVERLAY_AREA_PTR, _PROGRESS_OVERLAY_REGION_PTR
    global _PROGRESS_OVERLAY_X, _PROGRESS_OVERLAY_Y

    area = getattr(context, "area", None)
    region = getattr(context, "region", None)
    window_region = None
    if area and area.type == 'VIEW_3D' and region:
        window_region = next((item for item in area.regions if item.type == 'WINDOW'), None)

    if not area or area.type != 'VIEW_3D' or not region or not window_region:
        area, window_region = _find_progress_overlay_region(context)

    if not area or not window_region:
        return

    _PROGRESS_OVERLAY_AREA_PTR = area.as_pointer()
    _PROGRESS_OVERLAY_REGION_PTR = window_region.as_pointer()

    if event and hasattr(event, "mouse_region_x") and hasattr(event, "mouse_region_y"):
        mouse_x = event.mouse_region_x + region.x - window_region.x
        mouse_y = event.mouse_region_y + region.y - window_region.y
    else:
        mouse_x = int(window_region.width * 0.5)
        mouse_y = int(window_region.height * 0.5)

    _PROGRESS_OVERLAY_X = max(24, min(window_region.width - 48, int(mouse_x + 18)))
    _PROGRESS_OVERLAY_Y = max(24, min(window_region.height - 24, int(mouse_y - 18)))

def _get_progress_window(context):
    window = getattr(context, "window", None) if context else None
    if window:
        return window
    fallback_context = getattr(bpy, "context", None)
    return getattr(fallback_context, "window", None) if fallback_context else None

def _find_progress_overlay_region(context):
    window = _get_progress_window(context)
    screen = getattr(window, "screen", None) if window else getattr(context, "screen", None)
    if not screen:
        return None, None

    for area in screen.areas:
        if area.type != 'VIEW_3D':
            continue
        window_region = next((item for item in area.regions if item.type == 'WINDOW'), None)
        if window_region:
            return area, window_region

    return None, None

def _progress_set_label(label):
    global _PROGRESS_LABEL

    _PROGRESS_LABEL = label or "Working..."
    _redraw_progress_ui()

def _progress_prepare(context, label="Working..."):
    global _WAIT_CURSOR_WINDOW, _PROGRESS_TOTAL, _PROGRESS_VALUE

    if not _PROGRESS_OVERLAY_AREA_PTR or not _PROGRESS_OVERLAY_REGION_PTR:
        _remember_progress_anchor(context)

    _PROGRESS_TOTAL = 100.0
    _PROGRESS_VALUE = 0.0
    _progress_set_label(label)

    if _PROGRESS_OVERLAY_AREA_PTR and _PROGRESS_OVERLAY_REGION_PTR:
        _ensure_progress_overlay_handler()

    window = _get_progress_window(context)
    if window:
        try:
            window.cursor_set('WAIT')
            _WAIT_CURSOR_WINDOW = window
        except Exception:
            _WAIT_CURSOR_WINDOW = None

    _redraw_progress_ui()

def _reset_progress_feedback(context=None):
    global _WAIT_CURSOR_DEPTH, _WAIT_CURSOR_WINDOW
    global _PROGRESS_TOTAL, _PROGRESS_VALUE, _PROGRESS_LABEL
    global _PROGRESS_OVERLAY_AREA_PTR, _PROGRESS_OVERLAY_REGION_PTR

    window = _WAIT_CURSOR_WINDOW or _get_progress_window(context)
    if window:
        try:
            window.cursor_set('DEFAULT')
        except Exception:
            pass

    _WAIT_CURSOR_DEPTH = 0
    _WAIT_CURSOR_WINDOW = None
    _PROGRESS_TOTAL = 1.0
    _PROGRESS_VALUE = 0.0
    _PROGRESS_LABEL = ""
    _PROGRESS_OVERLAY_AREA_PTR = None
    _PROGRESS_OVERLAY_REGION_PTR = None
    _remove_progress_overlay_handler()

def _progress_begin(context, total):
    global _WAIT_CURSOR_DEPTH, _WAIT_CURSOR_WINDOW
    global _PROGRESS_TOTAL, _PROGRESS_VALUE

    if not _PROGRESS_OVERLAY_AREA_PTR or not _PROGRESS_OVERLAY_REGION_PTR:
        _remember_progress_anchor(context)

    window = _get_progress_window(context)
    if window and _WAIT_CURSOR_DEPTH == 0:
        try:
            window.cursor_set('WAIT')
            _WAIT_CURSOR_WINDOW = window
        except Exception:
            _WAIT_CURSOR_WINDOW = None
        _PROGRESS_TOTAL = max(float(total), 1.0)
        _PROGRESS_VALUE = 0.0
    if _PROGRESS_OVERLAY_AREA_PTR and _PROGRESS_OVERLAY_REGION_PTR:
        _ensure_progress_overlay_handler()
    _WAIT_CURSOR_DEPTH += 1
    _redraw_progress_ui()

def _progress_update(context, value):
    global _PROGRESS_VALUE

    _PROGRESS_VALUE = max(0.0, min(float(value), _PROGRESS_TOTAL))
    _redraw_progress_ui()

def _progress_end(context):
    global _WAIT_CURSOR_DEPTH, _WAIT_CURSOR_WINDOW

    if _WAIT_CURSOR_DEPTH > 0:
        _WAIT_CURSOR_DEPTH -= 1

    if _WAIT_CURSOR_DEPTH == 0:
        _reset_progress_feedback(context)
    _redraw_progress_ui()

def _start_deferred_task(operator, context, label="Working..."):
    wm = getattr(context, "window_manager", None)
    window = _get_progress_window(context)
    if not wm or not window:
        return {'CANCELLED'}

    operator._cp2pbr_started = False
    operator._cp2pbr_timer = wm.event_timer_add(0.01, window=window)
    _progress_prepare(context, label)
    wm.modal_handler_add(operator)
    return {'RUNNING_MODAL'}

def _finish_deferred_task(operator, context):
    wm = getattr(context, "window_manager", None)
    timer = getattr(operator, "_cp2pbr_timer", None)
    if wm and timer:
        try:
            wm.event_timer_remove(timer)
        except Exception:
            pass
    operator._cp2pbr_timer = None

    if _WAIT_CURSOR_DEPTH == 0:
        _reset_progress_feedback(context)

def _handle_deferred_modal(operator, context, event, callback):
    if event.type == 'ESC' and not getattr(operator, "_cp2pbr_started", False):
        _finish_deferred_task(operator, context)
        return {'CANCELLED'}

    if event.type != 'TIMER':
        return {'PASS_THROUGH'}

    if getattr(operator, "_cp2pbr_started", False):
        return {'RUNNING_MODAL'}

    operator._cp2pbr_started = True
    try:
        result = callback(context)
    except Exception as exc:
        operator.report({'ERROR'}, str(exc))
        result = {'CANCELLED'}
    finally:
        _finish_deferred_task(operator, context)

    return result

# ----------------------------
# Update Callbacks for Ramps
# ----------------------------

def _safe_set_ramp(cr, pos_black, pos_grey_rel, pos_white, col_b=(0,0,0,1), col_w=(1,1,1,1)):
    b = min(max(pos_black, 0.0), 1.0)
    w = min(max(pos_white, 0.0), 1.0)
    
    if b >= w:
        b = max(0.0, w - 0.001)
        if b == 0.0: w = 0.001
        
    g_rel = min(max(pos_grey_rel, 0.01), 0.99)
    g_abs = b + g_rel * (w - b)
    
    col_g = (
        (col_b[0] + col_w[0]) / 2.0,
        (col_b[1] + col_w[1]) / 2.0,
        (col_b[2] + col_w[2]) / 2.0,
        1.0
    )
    
    while len(cr.elements) > 1:
        cr.elements.remove(cr.elements[-1])
        
    cr.elements[0].position = b
    cr.elements[0].color = col_b
    
    e1 = cr.elements.new(g_abs)
    e1.color = col_g
    
    e2 = cr.elements.new(w)
    e2.color = col_w

_DEBUG_NODE_NAMES = (
    "CP2PBR_Debug_Compare0",
    "CP2PBR_Debug_Compare1",
    "CP2PBR_Debug_Mix0",
    "CP2PBR_Debug_Mix1",
)

def _remove_clipping_debug_nodes(nodes):
    for nm in _DEBUG_NODE_NAMES:
        n = nodes.get(nm)
        if n:
            nodes.remove(n)

def _restore_material_output_to_bsdf(node_tree, bsdf, out_node, settings=None):
    nodes = node_tree.nodes
    links = node_tree.links
    preview_node = nodes.get("CP2PBR_Preview_Emission")
    if preview_node:
        _remove_node_and_links(node_tree, preview_node)
    _remove_clipping_debug_nodes(nodes)
    _remove_hist_nodes(nodes)
    if settings is not None:
        settings.current_preview_map = "NONE"
    if out_node.inputs["Surface"].is_linked:
        for link in list(out_node.inputs["Surface"].links):
            links.remove(link)
    links.new(bsdf.outputs[0], out_node.inputs["Surface"])
    return bsdf.outputs[0]

def _ensure_clipping_debug_chain(node_tree, src_socket, epsilon=0.001):
    nodes = node_tree.nodes
    links = node_tree.links
    
    out_loc_x = 0.0
    out_loc_y = 0.0
    out_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
    if out_node:
        out_loc_x, out_loc_y = out_node.location.x, out_node.location.y
        
    comp0 = nodes.get("CP2PBR_Debug_Compare0")
    if not comp0:
        comp0 = nodes.new("ShaderNodeMath")
        comp0.name = comp0.label = "CP2PBR_Debug_Compare0"
        comp0.operation = 'COMPARE'
        comp0.location = (out_loc_x - 900, out_loc_y + 220)
    comp0.inputs[1].default_value = 0.0
    comp0.inputs[2].default_value = epsilon
        
    comp1 = nodes.get("CP2PBR_Debug_Compare1")
    if not comp1:
        comp1 = nodes.new("ShaderNodeMath")
        comp1.name = comp1.label = "CP2PBR_Debug_Compare1"
        comp1.operation = 'COMPARE'
        comp1.location = (out_loc_x - 900, out_loc_y + 40)
    comp1.inputs[1].default_value = 1.0
    comp1.inputs[2].default_value = epsilon
        
    mix0 = nodes.get("CP2PBR_Debug_Mix0")
    if not mix0:
        mix0 = nodes.new("ShaderNodeMixRGB")
        mix0.name = mix0.label = "CP2PBR_Debug_Mix0"
        mix0.blend_type = 'MIX'
        mix0.location = (out_loc_x - 700, out_loc_y + 130)
    mix0.inputs[2].default_value = (0.0, 0.0, 1.0, 1.0) # BLUE for Black clipping
        
    mix1 = nodes.get("CP2PBR_Debug_Mix1")
    if not mix1:
        mix1 = nodes.new("ShaderNodeMixRGB")
        mix1.name = mix1.label = "CP2PBR_Debug_Mix1"
        mix1.blend_type = 'MIX'
        mix1.location = (out_loc_x - 470, out_loc_y + 130)
    mix1.inputs[2].default_value = (1.0, 0.0, 0.0, 1.0) # RED for White clipping
        
    def _unlink_input(inp):
        if inp.is_linked:
            for lk in list(inp.links):
                links.remove(lk)
                
    _unlink_input(comp0.inputs[0])
    _unlink_input(comp1.inputs[0])
    links.new(src_socket, comp0.inputs[0])
    links.new(src_socket, comp1.inputs[0])
    
    _unlink_input(mix0.inputs[0]) 
    _unlink_input(mix0.inputs[1]) 
    links.new(comp0.outputs[0], mix0.inputs[0])
    links.new(src_socket, mix0.inputs[1])
    
    _unlink_input(mix1.inputs[0]) 
    _unlink_input(mix1.inputs[1]) 
    links.new(comp1.outputs[0], mix1.inputs[0])
    links.new(mix0.outputs[0], mix1.inputs[1])
    
    return mix1.outputs[0]

def update_clipping_preview(self, context):
    s = context.scene.cp2pbr_settings
    obj = _get_input_mesh(context)
    if not obj or not obj.active_material or not obj.active_material.use_nodes: return
    nodes = obj.active_material.node_tree.nodes
    links = obj.active_material.node_tree.links

    preview_node = nodes.get("CP2PBR_Preview_Emission")
    if not preview_node: return

    map_type = getattr(s, "current_preview_map", "NONE")
    if map_type not in ['ROUGHNESS', 'METALNESS']: return

    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not bsdf: return
    target_input = "Roughness" if map_type == 'ROUGHNESS' else "Metallic"
    if not bsdf.inputs[target_input].is_linked: return
    src_socket = bsdf.inputs[target_input].links[0].from_socket

    if s.clipping_preview:
        dbg_out = _ensure_clipping_debug_chain(obj.active_material.node_tree, src_socket, epsilon=0.001)
        if preview_node.inputs["Color"].is_linked:
            for lk in list(preview_node.inputs["Color"].links): links.remove(lk)
        links.new(dbg_out, preview_node.inputs["Color"])
    else:
        _remove_clipping_debug_nodes(nodes)
        if preview_node.inputs["Color"].is_linked:
            for lk in list(preview_node.inputs["Color"].links): links.remove(lk)
        links.new(src_socket, preview_node.inputs["Color"])

def update_metal_hsv(self, context):
    obj = _get_input_mesh(context)
    if not obj or not obj.active_material: return
    mat = obj.active_material
    if mat.use_nodes:
        nodes = mat.node_tree.nodes
        if "CP2PBR_MetalHSV" in nodes:
            hsv = nodes["CP2PBR_MetalHSV"]
            hsv.inputs["Hue"].default_value = self.metal_hsv_hue
            hsv.inputs["Saturation"].default_value = self.metal_hsv_saturation
            hsv.inputs["Value"].default_value = self.metal_hsv_value

def update_fake_normal(self, context):
    obj = _get_input_mesh(context)
    if not obj or not obj.active_material or not obj.active_material.use_nodes:
        return

    node_tree = obj.active_material.node_tree
    nodes = node_tree.nodes
    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not bsdf:
        return

    cp_node = nodes.get(_CP_TEXTURE_NODE_NAME)
    cp_socket = cp_node.outputs.get("Color") if cp_node else None
    _configure_fake_normal(node_tree, bsdf, cp_socket, self.enable_fake_normal_map, self.fake_normal_distance)

def update_shader_ramps(self, context):
    obj = _get_input_mesh(context)
    if not obj or not obj.active_material: return
    mat = obj.active_material
    if mat.use_nodes:
        nodes = mat.node_tree.nodes
        if "CP2PBR_RoughnessRamp" in nodes:
            cr = nodes["CP2PBR_RoughnessRamp"].color_ramp
            _safe_set_ramp(cr, 1.0 - self.roughness_black, 1.0 - self.roughness_grey, 1.0 - self.roughness_white,
                           self.roughness_black_col, self.roughness_white_col)
        if "CP2PBR_MetalnessRamp" in nodes:
            cm = nodes["CP2PBR_MetalnessRamp"].color_ramp
            _safe_set_ramp(cm, self.metalness_black, self.metalness_grey, self.metalness_white,
                           self.metalness_black_col, self.metalness_white_col)

def _hist_node_name(kind, part):
    return f"CP2PBR_Hist_{kind.upper()}_{part}"

_HIST_NODE_PARTS = ("Tex", "Sub", "Div", "Mul", "Add")

def _update_hist_chain_values(nodes, settings, kind):
    black, white, black_col, white_col = _hist_params(settings, kind)
    span = max(white - black, 1e-4)
    sub = nodes.get(_hist_node_name(kind, "Sub"))
    div = nodes.get(_hist_node_name(kind, "Div"))
    mul = nodes.get(_hist_node_name(kind, "Mul"))
    add = nodes.get(_hist_node_name(kind, "Add"))
    if sub: sub.inputs[2].default_value = (black, black, black, 1.0)
    if div: div.inputs[2].default_value = (span, span, span, 1.0)
    if mul: mul.inputs[2].default_value = tuple(w - b for b, w in zip(black_col, white_col)) + (1.0,)
    if add: add.inputs[2].default_value = tuple(black_col) + (1.0,)

def _remove_hist_nodes(nodes, keep=None):
    for kind in _HIST_KINDS:
        if kind == keep:
            continue
        for part in _HIST_NODE_PARTS:
            n = nodes.get(_hist_node_name(kind, part))
            if n:
                nodes.remove(n)

def _ensure_hist_chain(node_tree, settings, kind, image, location):
    """Build image -> (c-min)/(max-min) -> lerp(black_col, white_col); returns the output socket."""
    nodes, links = node_tree.nodes, node_tree.links
    x, y = location

    def get_node(part, node_type, offset):
        n = nodes.get(_hist_node_name(kind, part))
        if n and n.bl_idname != node_type:
            nodes.remove(n)
            n = None
        if not n:
            n = nodes.new(node_type)
            n.name = n.label = _hist_node_name(kind, part)
            n.location = (x + offset, y)
        return n

    tex = get_node("Tex", "ShaderNodeTexImage", 0)
    tex.image = image
    _set_colorspace(tex.image, "sRGB")

    chain = [tex]
    for part, blend, offset, clamp in (
        ("Sub", 'SUBTRACT', 300, False),
        ("Div", 'DIVIDE', 500, True),
        ("Mul", 'MULTIPLY', 700, False),
        ("Add", 'ADD', 900, True),
    ):
        node = get_node(part, "ShaderNodeMixRGB", offset)
        node.blend_type = blend
        node.use_clamp = clamp
        node.inputs[0].default_value = 1.0
        for lk in list(node.inputs[1].links):
            links.remove(lk)
        links.new(chain[-1].outputs[0], node.inputs[1])
        chain.append(node)

    _update_hist_chain_values(nodes, settings, kind)
    return chain[-1].outputs[0]

def update_hist_preview(self, context):
    obj = _get_input_mesh(context)
    if not obj or not obj.active_material or not obj.active_material.use_nodes:
        return
    nodes = obj.active_material.node_tree.nodes
    for kind in _HIST_KINDS:
        if nodes.get(_hist_node_name(kind, "Add")):
            _update_hist_chain_values(nodes, self, kind)

# ----------------------------
# Properties
# ----------------------------

class CP2PBR_Settings(PropertyGroup):
    """Property group for CP2PBR addon settings."""
    np_path: StringProperty(name="Non-Cross-Polarized", subtype="FILE_PATH", default="//import non-cross-polarized texture")
    cp_path: StringProperty(name="Cross-Polarized", subtype="FILE_PATH", default="//import cross-polarized texture")
    output_dir: StringProperty(name="Output folder", subtype="DIR_PATH", default="//set output folder")
    input_mesh: PointerProperty(name="Input Mesh", type=bpy.types.Object, poll=_mesh_object_poll)
    
    export_format: EnumProperty(
        name="Export Format",
        items=[
            ('PNG', "PNG", "Save as PNG"),
            ('TIFF', "TIFF", "Save as TIFF (.tif)"),
            ('JPEG', "JPG", "Save as JPEG (.jpg)")
        ],
        default='PNG'
    )
    
    cp_brightness_input: FloatProperty(name="adjust input CP brightness", default=-0.0, min=-1.0, max=1.0, step=1, precision=3)
    debug_weight_r: FloatProperty(name="Red weight", default=0.2126729, min=0.0, max=100.0, precision=4)
    debug_weight_g: FloatProperty(name="Green weight", default=0.7151522, min=0.0, max=100.0, precision=4)
    debug_weight_b: FloatProperty(name="Blue weight", default=0.0721750, min=0.0, max=100.0, precision=4)
    
    roughness_black_col: FloatVectorProperty(name="Black Color", subtype='COLOR', size=4, default=(1.0, 1.0, 1.0, 1.0), min=0.0, max=1.0, update=update_shader_ramps)
    roughness_white_col: FloatVectorProperty(name="White Color", subtype='COLOR', size=4, default=(0.0, 0.0, 0.0, 1.0), min=0.0, max=1.0, update=update_shader_ramps)
    
    metalness_black_col: FloatVectorProperty(name="Black Color", subtype='COLOR', size=4, default=(0.0, 0.0, 0.0, 1.0), min=0.0, max=1.0, update=update_shader_ramps)
    metalness_white_col: FloatVectorProperty(name="White Color", subtype='COLOR', size=4, default=(1.0, 1.0, 1.0, 1.0), min=0.0, max=1.0, update=update_shader_ramps)
    
    current_preview_map: StringProperty(default="NONE", options={"HIDDEN"})
    clipping_preview: BoolProperty(name="clipping preview", default=False, update=update_clipping_preview)
    roughness_white: FloatProperty(name="Roughness Min", default=0.0, min=0.0, max=1.0, update=update_shader_ramps)
    roughness_grey: FloatProperty(name="Roughness Blend", default=0.5, min=0.01, max=0.99, update=update_shader_ramps)
    roughness_black: FloatProperty(name="Roughness Max", default=1.0, min=0.0, max=1.0, update=update_shader_ramps)

    
    metalness_black: FloatProperty(name="Metalness Min", default=0.0, min=0.0, max=1.0, update=update_shader_ramps)
    metalness_grey: FloatProperty(name="Metalness Blend", default=0.5, min=0.01, max=0.99, update=update_shader_ramps)
    metalness_white: FloatProperty(name="Metalness Max", default=1.0, min=0.0, max=1.0, update=update_shader_ramps)

    metal_hsv_hue: FloatProperty(name="Metal Hue", default=0.5, min=0.0, max=1.0, update=update_metal_hsv)
    metal_hsv_saturation: FloatProperty(name="Metal Saturation", default=1.0, min=0.0, max=2.0, update=update_metal_hsv)
    metal_hsv_value: FloatProperty(name="Metal Value", default=1.0, min=0.0, max=2.0, update=update_metal_hsv)
    enable_fake_normal_map: BoolProperty(name="Fake normal map", default=False, update=update_fake_normal)
    fake_normal_distance: FloatProperty(name="Distance", default=0.001, min=0.0, max=0.1, precision=4, update=update_fake_normal)
    custom_normal_map_path: StringProperty(
        name="Custom normal map",
        description="Add custom normal map",
        subtype="FILE_PATH",
        default=""
    )
    
    show_adjust_cp: BoolProperty(name="Adjust CP Brightness Options", default=False)
    show_advanced_debug: BoolProperty(name="Advanced", default=False)
    export_rgb_diff_maps: BoolProperty(
        name="Export RGB difference maps",
        description="Also export greyscale |NPC - CP| maps for the R, G and B channels and their sum",
        default=False,
    )
    np_hist_black: FloatProperty(name="NPC Min", default=0.0, min=0.0, max=1.0, precision=3, update=update_hist_preview)
    np_hist_white: FloatProperty(name="NPC Max", default=1.0, min=0.0, max=1.0, precision=3, update=update_hist_preview)
    np_hist_black_col: FloatVectorProperty(name="NPC Min Color", subtype='COLOR', size=4, default=(0.0, 0.0, 0.0, 1.0), min=0.0, max=1.0, update=update_hist_preview)
    np_hist_white_col: FloatVectorProperty(name="NPC Max Color", subtype='COLOR', size=4, default=(1.0, 1.0, 1.0, 1.0), min=0.0, max=1.0, update=update_hist_preview)
    cp_hist_black: FloatProperty(name="CP Min", default=0.0, min=0.0, max=1.0, precision=3, update=update_hist_preview)
    cp_hist_white: FloatProperty(name="CP Max", default=1.0, min=0.0, max=1.0, precision=3, update=update_hist_preview)
    cp_hist_black_col: FloatVectorProperty(name="CP Min Color", subtype='COLOR', size=4, default=(0.0, 0.0, 0.0, 1.0), min=0.0, max=1.0, update=update_hist_preview)
    cp_hist_white_col: FloatVectorProperty(name="CP Max Color", subtype='COLOR', size=4, default=(1.0, 1.0, 1.0, 1.0), min=0.0, max=1.0, update=update_hist_preview)
    reuse_existing_debug_maps: BoolProperty(name="Reuse existing debug maps", default=True)
    reuse_histogram_params_from_file: BoolProperty(name="Load normalization preset (txt)", default=False)
    enable_normalization: BoolProperty(
        name="Normalization",
        description="Normalize the debug subtraction maps to the 0-1 range",
        default=True,
    )
    histogram_params_file: StringProperty(name="Histogram params file", subtype="FILE_PATH", default="")
    reuse_existing_baked_maps: BoolProperty(name="Reuse previously baked maps if available", default=True)
    
    enable_resize: BoolProperty(name="Reduce Resolution", default=True)
    target_width: bpy.props.IntProperty(name="Max Width", default=2048, min=1, subtype='PIXEL')
    
    show_optional_blur: BoolProperty(name="Enable Optional Blur", default=False)
    blur_metalness: bpy.props.FloatProperty(name="Blur Metalness (%)", default=0.00, min=0.0, max=100.0)
    blur_roughness: bpy.props.FloatProperty(name="Blur Roughness (%)", default=0.00, min=0.0, max=100.0)
    erode_metalness_clusters: BoolProperty(
        name="Remove small metallic islands",
        description="Turn grey/white clusters of pixels smaller than the threshold black in the exported metalness map",
        default=False,
    )
    erode_cluster_threshold: bpy.props.FloatProperty(
        name="Island size (%)",
        description="Islands smaller than a square whose side is this percentage of the image width are turned black (100% = width x width pixels)",
        default=3.0, min=0.0, max=100.0, precision=2,
    )

# ----------------------------
# Operators
# ----------------------------

class CP2PBR_OT_extract_lightness(Operator):
    """Operator to extract lightness from NP and CP textures."""
    bl_idname = "cp2pbr.extract_lightness"
    bl_label = "Extract lightness"
    show_progress: BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, context, event):
        if self.show_progress:
            _remember_progress_anchor(context, event)
            return _start_deferred_task(self, context, "Extracting light textures")
        return self._execute_sync(context)

    def modal(self, context, event):
        return _handle_deferred_modal(self, context, event, self._execute_sync)

    def execute(self, context):
        if self.show_progress:
            _remember_progress_anchor(context)
            return _start_deferred_task(self, context, "Extracting light textures")
        return self._execute_sync(context)

    def _execute_sync(self, context):
        s = context.scene.cp2pbr_settings
        out_dir = _norm_dir(s.output_dir)
        debug_dir = os.path.join(out_dir, "debug maps") if out_dir else ""
        out_dir = debug_dir
        
        if not s.np_path or not s.cp_path:
            self.report({"ERROR"}, "Please import both Non-Cross-Polarized and Cross-Polarized textures first.")
            return {"CANCELLED"}
        if not out_dir:
            self.report({"ERROR"}, "Please set an output folder first.")
            return {"CANCELLED"}
        _ensure_dir(out_dir)

        ok, msg = _export_adjusted_textures(context, s)
        if not ok:
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        np_img = _load_image_raw(_source_texture_path(s, "np"))
        cp_img = _load_image_raw(_source_texture_path(s, "cp"))
        if np_img is None or cp_img is None:
            _discard_image(np_img)
            _discard_image(cp_img)
            self.report({"ERROR"}, "Could not load input textures.")
            return {"CANCELLED"}
            
        weights = _normalized_rgb_weights(s)

        if self.show_progress:
            _progress_begin(context, 2.0)
        try:
            if self.show_progress:
                _progress_set_label("Extracting NP light")
                _progress_update(context, 0.2)
            _extract_light_image_to_file(context, np_img, _np_light_path(s), ".tif", weights)
            if self.show_progress:
                _progress_set_label("Extracting CP light")
                _progress_update(context, 1.2)
            _extract_light_image_to_file(context, cp_img, _cp_light_path(s), ".tif", weights)
            if self.show_progress:
                _progress_set_label("Light extraction complete")
                _progress_update(context, 2.0)
        except MemoryError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        finally:
            _discard_image(np_img)
            _discard_image(cp_img)
            if self.show_progress:
                _progress_end(context)
        return {"FINISHED"}

def _export_normalized_difference(context, out_name, brightness_adj, show_progress=True):
    s = context.scene.cp2pbr_settings
    out_dir = _norm_dir(s.output_dir)
    if out_dir:
        out_dir = os.path.join(out_dir, "debug maps")
        _ensure_dir(out_dir)
    if not out_dir: return (False, "Please set an output folder.")
    np_light, cp_light = _guess_map_path(s, "NP_Light"), _guess_map_path(s, "CP_Light")
    if not os.path.isfile(np_light) or not os.path.isfile(cp_light): return (False, "NP/CP Light not found.")

    np_img, cp_img = _load_image_raw(np_light), _load_image_raw(cp_light)
    if not np_img or not cp_img:
        _discard_image(np_img)
        _discard_image(cp_img)
        return (False, "Could not load NP/CP Light.")
    w, h = np_img.size[0], np_img.size[1]
    ext = ".tif"
    factor = _safe_factor(brightness_adj)
    histogram_params = None
    normalize = bool(s.enable_normalization)
    if normalize and s.reuse_histogram_params_from_file:
        loaded_params, loaded_msg = _load_histogram_params_file(s.histogram_params_file)
        if not loaded_params:
            _discard_image(np_img)
            _discard_image(cp_img)
            return (False, loaded_msg)
        histogram_params = loaded_params

    if show_progress:
        _progress_begin(context, 4.0)
    try:
        if show_progress:
            _progress_set_label("Loading light maps")
            _progress_update(context, 0.5)
        if _HAS_NUMPY:
            buf = np.empty(w * h * 4, dtype=np.float32)
            if hasattr(np_img.pixels, "foreach_get"): np_img.pixels.foreach_get(buf)
            else: buf[:] = np.array(np_img.pixels[:], dtype=np.float32)
            base = np.clip(buf[0::4].astype(np.float32), 0.0, 1.0)
            if show_progress:
                _progress_set_label("Comparing NP and CP light")
                _progress_update(context, 1.5)

            if hasattr(cp_img.pixels, "foreach_get"): cp_img.pixels.foreach_get(buf)
            else: buf[:] = np.array(cp_img.pixels[:], dtype=np.float32)
            blend = np.clip(buf[0::4].astype(np.float32) * factor, 0.0, 1.0)
            del buf

            diff = base - blend
            auto_d_min, auto_d_max = float(diff.min()), float(diff.max())
            d_min = histogram_params["debug_diff_min"] if histogram_params else auto_d_min
            d_max = histogram_params["debug_diff_max"] if histogram_params else auto_d_max
            if not normalize:
                d_min, d_max = 0.0, 1.0
            if d_max > d_min:
                diff = (diff - d_min) / (d_max - d_min)
            else:
                diff = np.zeros_like(diff)
            diff = np.clip(diff, 0.0, 1.0).astype(np.float32)
            del base, blend

            out_buf = np.empty(w * h * 4, dtype=np.float32)
            out_buf[0::4] = diff
            out_buf[1::4] = diff
            out_buf[2::4] = diff
            out_buf[3::4] = 1.0
            del diff
        else:
            np_buf, cp_buf = _read_pixels_fast_fallback(np_img), _read_pixels_fast_fallback(cp_img)
            out_buf = array('f', [0.0]) * (w * h * 4)
            vals = []
            if show_progress:
                _progress_set_label("Comparing NP and CP light")
                _progress_update(context, 1.5)
            for i in range(0, len(np_buf), 4):
                val = _clamp01(np_buf[i]) - _clamp01(_clamp01(cp_buf[i]) * factor)
                vals.append(val)
            auto_d_min, auto_d_max = float(min(vals)), float(max(vals))
            d_min = histogram_params["debug_diff_min"] if histogram_params else auto_d_min
            d_max = histogram_params["debug_diff_max"] if histogram_params else auto_d_max
            if not normalize:
                d_min, d_max = 0.0, 1.0
            range_val = d_max - d_min if d_max > d_min else 1.0
            for i, val in enumerate(vals):
                d = _clamp01((val - d_min) / range_val) if d_max > d_min else 0.0
                idx = i * 4
                out_buf[idx:idx+4] = array('f', [d, d, d, 1.0])

        if show_progress:
            _progress_set_label("Writing debug map")
            _progress_update(context, 3.0)
        dst = _new_image_like(out_name, w, h, float_buffer=True, alpha=True)
        _set_colorspace(dst, "Non-Color")
        _write_pixels_fast(dst, out_buf)
        out_path = _build_out_path(out_dir, out_name, ext)
        try:
            _save_image_to(context, dst, out_path, tiff_16bit=True)
        finally:
            _discard_image(dst)
        channels_ok, channels_msg = _export_rgb_debug_channels(context, s, brightness_adj)
        if not channels_ok:
            return (False, channels_msg)
        params_ok, params_msg = _write_histogram_params_file(s, d_min, d_max)
        if not params_ok:
            return (False, params_msg)
        if show_progress:
            _progress_set_label("Debug map complete")
            _progress_update(context, 4.0)
        return (True, out_path)
    finally:
        _discard_image(np_img)
        _discard_image(cp_img)
        if show_progress:
            _progress_end(context)

class CP2PBR_OT_export_debug_map(Operator):
    bl_idname = "cp2pbr.export_debug_map"
    bl_label = "Export Debug Map"
    show_progress: BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, context, event):
        if self.show_progress:
            _remember_progress_anchor(context, event)
            return _start_deferred_task(self, context, "Generating debug map")
        return self._execute_sync(context)

    def modal(self, context, event):
        return _handle_deferred_modal(self, context, event, self._execute_sync)

    def execute(self, context):
        if self.show_progress:
            _remember_progress_anchor(context)
            return _start_deferred_task(self, context, "Generating debug map")
        return self._execute_sync(context)

    def _execute_sync(self, context):
        s = context.scene.cp2pbr_settings
        ok, msg = _export_normalized_difference(
            context,
            out_name="Debug_Subtr_Normalized",
            brightness_adj=s.cp_brightness_input,
            show_progress=self.show_progress,
        )
        if not ok: self.report({"ERROR"}, msg); return {"CANCELLED"}
        self.report({"INFO"}, f"Saved: {os.path.basename(msg)}")
        return {"FINISHED"}


# ----------------------------
# Map Preview Logic
# ----------------------------

class CP2PBR_OT_preview_map(Operator):
    bl_idname = "cp2pbr.preview_map"
    bl_label = "Preview Map"
    bl_options = {"REGISTER", "UNDO"}
    
    map_type: StringProperty(default='FINAL') # 'ROUGHNESS', 'METALNESS', 'ALBEDO', 'NORMAL', 'FINAL'
    
    def execute(self, context):
        s = context.scene.cp2pbr_settings
        obj = _require_input_mesh(context, self, "previewing a map")
        if not obj:
            return {"CANCELLED"}
        if self.map_type in _HIST_MAP_TYPES:
            mat = obj.active_material
            if not mat:
                mat = bpy.data.materials.new("CP2PBR_Shader")
                if obj.data.materials:
                    obj.data.materials[0] = mat
                else:
                    obj.data.materials.append(mat)
                obj.active_material = mat
            mat.use_nodes = True
            tree = mat.node_tree
            if not any(n.type == 'OUTPUT_MATERIAL' for n in tree.nodes):
                new_out = tree.nodes.new("ShaderNodeOutputMaterial")
                new_out.location = (300, 0)
            if not any(n.type == 'BSDF_PRINCIPLED' for n in tree.nodes):
                new_bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
                new_bsdf.location = (0, 0)
                out_existing = next(n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL')
                if not out_existing.inputs["Surface"].is_linked:
                    tree.links.new(new_bsdf.outputs[0], out_existing.inputs["Surface"])
        if not obj or not obj.active_material or not obj.active_material.use_nodes:
            self.report({"ERROR"}, "Input Mesh must have a valid node material.")
            return {"CANCELLED"}
            
        nodes = obj.active_material.node_tree.nodes
        links = obj.active_material.node_tree.links
        
        out_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        
        if not out_node or not bsdf:
            self.report({"ERROR"}, "Could not find Principled BSDF or Material Output.")
            return {"CANCELLED"}
            
        preview_node = nodes.get("CP2PBR_Preview_Emission")
        
        if self.map_type == 'FINAL':
            s.current_preview_map = 'NONE'
            if out_node.inputs["Surface"].is_linked:
                for lk in list(out_node.inputs["Surface"].links): links.remove(lk)
            links.new(bsdf.outputs[0], out_node.inputs["Surface"])
            if preview_node: nodes.remove(preview_node)
            _remove_clipping_debug_nodes(nodes)
            _remove_hist_nodes(nodes)
            return {"FINISHED"}

        s.current_preview_map = self.map_type
        _remove_hist_nodes(nodes, keep=_HIST_MAP_TYPES.get(self.map_type))

        if not preview_node:
            preview_node = nodes.new("ShaderNodeEmission")
            preview_node.name = "CP2PBR_Preview_Emission"
            preview_node.label = "Map Preview"
            preview_node.location = (out_node.location.x - 200, out_node.location.y + 200)

        if self.map_type in _HIST_MAP_TYPES:
            kind = _HIST_MAP_TYPES[self.map_type]
            source_path = s.np_path if kind == "np" else s.cp_path
            hist_img = _load_image(source_path)
            if not hist_img:
                self.report({"ERROR"}, f"Could not load the {_HIST_LABELS[kind]} texture.")
                return {"CANCELLED"}
            src_socket = _ensure_hist_chain(
                obj.active_material.node_tree, s, kind, hist_img,
                (out_node.location.x - 1500, out_node.location.y + (700 if kind == "np" else 1000)),
            )
        elif self.map_type == 'ALBEDO':
            base_color_input = bsdf.inputs.get("Base Color")
            if not base_color_input or not base_color_input.is_linked:
                self.report({"ERROR"}, "Base Color is not linked to any node.")
                return {"CANCELLED"}
            src_socket = base_color_input.links[0].from_socket
        elif self.map_type == 'NORMAL':
            normal_input = bsdf.inputs.get("Normal")
            if not normal_input or not normal_input.is_linked:
                self.report({"ERROR"}, "Normal is not linked to any node.")
                return {"CANCELLED"}
            src_socket = normal_input.links[0].from_socket
        else:
            target_input = "Roughness" if self.map_type == 'ROUGHNESS' else "Metallic"
            if not bsdf.inputs[target_input].is_linked:
                self.report({"ERROR"}, f"{target_input} is not linked to any map.")
                return {"CANCELLED"}
            src_socket = bsdf.inputs[target_input].links[0].from_socket

        use_clipping_preview = self.map_type in {'ROUGHNESS', 'METALNESS'} and s.clipping_preview
        if not use_clipping_preview:
            _remove_clipping_debug_nodes(nodes)
            
        if preview_node.inputs["Color"].is_linked:
            for lk in list(preview_node.inputs["Color"].links): links.remove(lk)
            
        if use_clipping_preview:
            dbg_out = _ensure_clipping_debug_chain(obj.active_material.node_tree, src_socket, epsilon=0.001)
            links.new(dbg_out, preview_node.inputs["Color"])
        else:
            links.new(src_socket, preview_node.inputs["Color"])
            
        if out_node.inputs["Surface"].is_linked:
            for lk in list(out_node.inputs["Surface"].links): links.remove(lk)
        links.new(preview_node.outputs[0], out_node.inputs["Surface"])
        
        self.report({"INFO"}, f"Previewing: {self.map_type}")
        return {"FINISHED"}

# ----------------------------
# Shaders and Math Processing
# ----------------------------

class CP2PBR_OT_create_apply_shader(Operator):
    bl_idname = "cp2pbr.create_apply_shader"
    bl_label = "Create & Apply Shader (raw maps)"
    
    def execute(self, context):
        s = context.scene.cp2pbr_settings
        obj = _require_input_mesh(context, self, "creating and applying the shader")
        if not obj:
            return {"CANCELLED"}

        debug_map_path = _guess_map_path(s, "Debug_Subtr_Normalized")
        if not debug_map_path:
            self.report({"ERROR"}, "Debug_Subtr_Normalized not found in the output folder.")
            return {"CANCELLED"}
        def _load_shader_texture(kind):
            path = _source_texture_path(s, kind)
            original = s.np_path if kind == "np" else s.cp_path
            if path != original:
                abs_path = os.path.normcase(os.path.abspath(bpy.path.abspath(path)))
                for stale in list(bpy.data.images):
                    if stale.filepath and os.path.normcase(os.path.abspath(bpy.path.abspath(stale.filepath))) == abs_path:
                        if stale.users == 0:
                            bpy.data.images.remove(stale)
                img = _load_image(path, force_reload=True)
                if img is not None:
                    try:
                        img.update()
                    except Exception:
                        pass
                    if img.size[0] > 0 and img.size[1] > 0:
                        return img
                path = original
            return _load_image(path, force_reload=True)

        np_img = _load_shader_texture("np")
        cp_img = _load_shader_texture("cp")
        r_img = _load_image(debug_map_path, force_reload=True)
        m_img = _load_image(debug_map_path, force_reload=True)
        if not np_img or not cp_img or not r_img or not m_img:
            self.report({"ERROR"}, "Could not load the source textures or the debug map.")
            return {"CANCELLED"}

        mat = obj.active_material
        if not mat:
            mat = bpy.data.materials.new("CP2PBR_Shader")
            if obj.data.materials: obj.data.materials[0] = mat
            else: obj.data.materials.append(mat)
            obj.active_material = mat

        mat.use_nodes = True
        nodes, links = mat.node_tree.nodes, mat.node_tree.links

        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        out_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)

        if not out_node:
            out_node = nodes.new("ShaderNodeOutputMaterial")
            out_node.location = (950, 0)
        if not bsdf:
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.location = (700, 0)
            if out_node.inputs[0]: links.new(bsdf.outputs[0], out_node.inputs[0])

        _restore_material_output_to_bsdf(mat.node_tree, bsdf, out_node, s)

        def delete_input_nodes(sock):
            if not sock.is_linked: return
            for link in list(sock.links):
                node = link.from_node
                links.remove(link)
                if not any(out.is_linked for out in node.outputs):
                    for inp in node.inputs:
                        delete_input_nodes(inp)
                    try: nodes.remove(node)
                    except: pass

        for sock_name in ["Base Color", "Roughness", "Metallic", "Normal"]:
            sock = bsdf.inputs.get(sock_name)
            if sock: delete_input_nodes(sock)

        loc_x, loc_y = bsdf.location.x, bsdf.location.y

        t_np = nodes.new("ShaderNodeTexImage")
        t_np.location, t_np.image = (loc_x - 700, loc_y + 550), np_img
        _set_colorspace(t_np.image, "sRGB")

        t_cp = nodes.new("ShaderNodeTexImage")
        t_cp.location, t_cp.image = (loc_x - 700, loc_y + 250), cp_img
        t_cp.name = t_cp.label = _CP_TEXTURE_NODE_NAME
        _set_colorspace(t_cp.image, "sRGB")

        t_r = nodes.new("ShaderNodeTexImage")
        t_r.location, t_r.image = (loc_x - 700, loc_y - 330), r_img
        _set_colorspace(t_r.image, "Non-Color")

        ramp_r = nodes.new("ShaderNodeValToRGB")
        ramp_r.name = ramp_r.label = "CP2PBR_RoughnessRamp"
        ramp_r.location = (loc_x - 340, loc_y - 330)
        _safe_set_ramp(ramp_r.color_ramp, 1.0 - s.roughness_black, 1.0 - s.roughness_grey, 1.0 - s.roughness_white, s.roughness_black_col, s.roughness_white_col)

        t_m = nodes.new("ShaderNodeTexImage")
        t_m.location, t_m.image = (loc_x - 700, loc_y - 50), m_img
        _set_colorspace(t_m.image, "Non-Color")

        ramp_m = nodes.new("ShaderNodeValToRGB")
        ramp_m.name = ramp_m.label = "CP2PBR_MetalnessRamp"
        ramp_m.location = (loc_x - 340, loc_y - 50)
        _safe_set_ramp(ramp_m.color_ramp, s.metalness_black, s.metalness_grey, s.metalness_white, s.metalness_black_col, s.metalness_white_col)

        # --- Metal-only HSV control on Base Color ---
        hsv = nodes.get("CP2PBR_MetalHSV")
        if not hsv:
            hsv = nodes.new("ShaderNodeHueSaturation")
            hsv.name = hsv.label = "CP2PBR_MetalHSV"
            hsv.location = (loc_x - 320, loc_y + 440)
        hsv.inputs["Hue"].default_value = s.metal_hsv_hue
        hsv.inputs["Saturation"].default_value = s.metal_hsv_saturation

        hsv.inputs["Value"].default_value = s.metal_hsv_value
        hsv.inputs["Fac"].default_value = 1.0

        mix = nodes.get("CP2PBR_MetalHSV_Mix")
        if not mix:
            mix = nodes.new("ShaderNodeMixRGB")
            mix.name = mix.label = "CP2PBR_MetalHSV_Mix"
            mix.blend_type = 'MIX'
            mix.location = (loc_x - 50, loc_y + 250)

        for link in list(hsv.inputs["Color"].links):
            links.remove(link)
        links.new(t_np.outputs["Color"], hsv.inputs["Color"])

        for link in list(mix.inputs[1].links):
            links.remove(link)
        for link in list(mix.inputs[2].links):
            links.remove(link)
        links.new(t_cp.outputs["Color"], mix.inputs[1])
        links.new(hsv.outputs["Color"], mix.inputs[2])

        for link in list(mix.inputs[0].links):
            links.remove(link)
        links.new(ramp_m.outputs["Color"], mix.inputs[0])

        if bsdf.inputs.get("Base Color"):
            for link in list(bsdf.inputs["Base Color"].links):
                links.remove(link)
            links.new(mix.outputs[0], bsdf.inputs["Base Color"])
        links.new(t_r.outputs["Color"], ramp_r.inputs["Fac"])
        if bsdf.inputs.get("Roughness"):
            links.new(ramp_r.outputs["Color"], bsdf.inputs["Roughness"])
        links.new(t_m.outputs["Color"], ramp_m.inputs["Fac"])
        if bsdf.inputs.get("Metallic"):
            links.new(ramp_m.outputs["Color"], bsdf.inputs["Metallic"])
        _configure_fake_normal(mat.node_tree, bsdf, t_cp.outputs["Color"], s.enable_fake_normal_map, s.fake_normal_distance)

        self.report({"INFO"}, "Shader created and applied from raw maps.")
        return {"FINISHED"}

class CP2PBR_OT_full_bake(Operator):
    bl_idname = "cp2pbr.full_bake"
    bl_label = "Bake Maps & Apply Shader"
    bl_options = {'REGISTER', 'UNDO'}
    overwrite_targets: StringProperty(default="", options={'HIDDEN'})
    reuse_missing_targets: StringProperty(default="", options={'HIDDEN'})

    def draw(self, context):
        layout = self.layout
        if self.reuse_missing_targets:
            layout.label(text="Previously baked maps were not found for all required channels.")
            layout.label(text="A new bake is required. Continue?")
            for name in [item for item in self.reuse_missing_targets.split("|") if item]:
                layout.label(text=name, icon='ERROR')
        else:
            layout.label(text="The following baked maps already exist and will be overwritten:")
            for name in [item for item in self.overwrite_targets.split("|") if item]:
                layout.label(text=name, icon='FILE_IMAGE')

    def invoke(self, context, event):
        _remember_progress_anchor(context, event)

        obj = _get_input_mesh(context)
        if not obj:
            return self.execute(context)

        mat = obj.active_material
        s = context.scene.cp2pbr_settings
        suffixes = _linked_bake_suffixes(mat)
        expected_paths = _expected_baked_map_paths(s, suffixes)
        existing_paths = [path for path in expected_paths.values() if os.path.isfile(path)]

        self.overwrite_targets = ""
        self.reuse_missing_targets = ""

        if s.reuse_existing_baked_maps:
            missing_suffixes = [suffix for suffix, path in expected_paths.items() if not os.path.isfile(path)]
            if not missing_suffixes:
                return self.execute(context)
            self.reuse_missing_targets = "|".join(missing_suffixes)
            return context.window_manager.invoke_props_dialog(self, width=360)

        if not existing_paths:
            return self.execute(context)

        self.overwrite_targets = "|".join(os.path.basename(path) for path in existing_paths)
        return context.window_manager.invoke_props_dialog(self, width=360)

    def modal(self, context, event):
        return _handle_deferred_modal(self, context, event, self._execute_sync)

    def execute(self, context):
        _remember_progress_anchor(context)
        return _start_deferred_task(self, context, "Preparing bake workflow")

    def _execute_sync(self, context):
        obj = _require_input_mesh(context, self, "baking maps and applying the shader")
        if not obj:
            return {'CANCELLED'}
        _sync_active_selected_object(context, obj)

        mat = obj.active_material
        if not mat or not mat.use_nodes:
            self.report({'ERROR'}, "Input Mesh must have a valid node material.")
            return {'CANCELLED'}

        s = context.scene.cp2pbr_settings
        outdir = getattr(s, "output_dir", "")
        if not outdir:
            self.report({'ERROR'}, "Please set an output folder first.")
            return {'CANCELLED'}

        suffixes = _linked_bake_suffixes(mat)
        existing_baked_paths = _expected_baked_map_paths(s, suffixes)
        can_reuse_baked = bool(
            s.reuse_existing_baked_maps and
            suffixes and
            all(os.path.isfile(path) for path in existing_baked_paths.values())
        )

        if can_reuse_baked:
            _progress_set_label("Applying previously baked maps")
            if not _apply_baked_maps_to_material(mat, existing_baked_paths):
                self.report({'ERROR'}, "Could not apply the previously baked maps.")
                return {'CANCELLED'}
            self.report({'INFO'}, "Previously baked maps found and applied.")
            return {'FINISHED'}

        processed_dir = os.path.join(bpy.path.abspath(outdir), "processed textures")
        os.makedirs(processed_dir, exist_ok=True)

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        out_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if not bsdf or not out_node:
            self.report({'ERROR'}, "Could not find Principled BSDF or Material Output.")
            return {'CANCELLED'}

        orig_surface_link = _restore_material_output_to_bsdf(mat.node_tree, bsdf, out_node, s)
        rebuilt_baked_material = False

        scn = context.scene
        old_engine = scn.render.engine
        old_view_transform = scn.view_settings.view_transform

        if not hasattr(scn, 'cycles'):
            self.report({'ERROR'}, "Cycles engine is not available.")
            return {'CANCELLED'}

        old_device = scn.cycles.device
        old_denoise = getattr(scn.cycles, 'use_denoising', False)
        old_samples = scn.cycles.samples

        scn.render.engine = 'CYCLES'
        scn.cycles.device = 'GPU'
        if hasattr(scn.cycles, 'use_denoising'):
            scn.cycles.use_denoising = False
        scn.cycles.samples = 1

        scn.view_settings.view_transform = 'Standard'

        def get_connected_image_node(socket):
            if not socket.is_linked: return None
            node = socket.links[0].from_node
            if node.type == 'TEX_IMAGE':
                return node
            if socket.name == 'Normal' and node.type == 'NORMAL_MAP' and node.inputs['Color'].is_linked:
                 tex_node = node.inputs['Color'].links[0].from_node
                 if tex_node.type == 'TEX_IMAGE': return tex_node
            for inp in node.inputs:
                res = get_connected_image_node(inp)
                if res: return res
            return None
            
        def is_direct_texture_connection(socket):
            if not socket.is_linked: return False, None
            if socket.name == 'Normal':
                return False, None
            node = socket.links[0].from_node
            if node.type == 'TEX_IMAGE':
                return True, node
            return False, None

        def get_resolution(socket):
            tex_node = get_connected_image_node(socket)
            if tex_node and tex_node.image:
                return _apply_resize_limit(s, tex_node.image.size[0], tex_node.image.size[1])
            
            base_col_sock = bsdf.inputs.get('Base Color')
            if base_col_sock:
                base_tex = get_connected_image_node(base_col_sock)
                if base_tex and base_tex.image:
                    return _apply_resize_limit(s, base_tex.image.size[0], base_tex.image.size[1])
                    
            target_w = getattr(s, "target_width", 2048) if getattr(s, "enable_resize", False) else 2048
            return _apply_resize_limit(s, target_w, target_w)

        channels = [
            ('Base Color', 'EMIT', 'BaseColor', False),
            ('Metallic', 'EMIT', 'Metalness', True),
            ('Roughness', 'EMIT', 'Roughness', True),
            ('Normal', 'NORMAL', 'Normal', True)
        ]

        emit_node = nodes.new('ShaderNodeEmission')
        emit_node.name = "CP2PBR_Bake_Emission"
        emit_node.location = (bsdf.location.x, bsdf.location.y - 300)

        normal_source_material = None
        normal_socket = bsdf.inputs.get("Normal")
        normal_direct_image_node = _get_direct_normal_image_node(normal_socket)
        if normal_socket and normal_socket.is_linked and normal_direct_image_node is None:
            normal_source_material = mat.copy()
            normal_source_material.name = f"{mat.name}_CP2PBR_NormalBakeSource"

        obj.select_set(True)
        context.view_layer.objects.active = obj

        def assign_active_material(material):
            if obj.material_slots:
                obj.material_slots[obj.active_material_index].material = material
            elif obj.data.materials:
                obj.data.materials[0] = material
            else:
                obj.data.materials.append(material)

        def bake_normal_isolated(outpath, width, height, ext_string):
            if normal_source_material is None:
                return False

            temp_nodes = normal_source_material.node_tree.nodes
            temp_links = normal_source_material.node_tree.links
            temp_bsdf = next((node for node in temp_nodes if node.type == 'BSDF_PRINCIPLED'), None)
            temp_out = next((node for node in temp_nodes if node.type == 'OUTPUT_MATERIAL'), None)
            if not temp_bsdf or not temp_out:
                raise RuntimeError("Could not build a temporary material for normal baking.")

            _restore_material_output_to_bsdf(normal_source_material.node_tree, temp_bsdf, temp_out)
            for socket_name in ["Base Color", "Roughness", "Metallic"]:
                _clear_bsdf_input(temp_nodes, temp_links, temp_bsdf, socket_name)

            img_name = f"{obj.name}_Normal_baked"
            bake_img = bpy.data.images.get(img_name)
            if bake_img:
                bpy.data.images.remove(bake_img)

            bake_img = bpy.data.images.new(name=img_name, width=width, height=height, alpha=False, float_buffer=True)
            try:
                bake_img.colorspace_settings.name = 'Non-Color'
            except Exception:
                pass

            bake_node = temp_nodes.new('ShaderNodeTexImage')
            bake_node.image = bake_img
            try:
                bake_node.image.colorspace_settings.name = 'Non-Color'
            except Exception:
                pass

            temp_nodes.active = bake_node
            bake_node.select = True
            for node in temp_nodes:
                if node != bake_node:
                    node.select = False

            original_material = obj.active_material
            assign_active_material(normal_source_material)
            obj.active_material = normal_source_material
            try:
                bpy.ops.object.bake(type='NORMAL', normal_space='TANGENT', use_clear=True, margin=16)
                bake_img.filepath_raw = outpath
                bake_img.file_format = ext_string if ext_string in ['PNG', 'JPEG', 'TIFF'] else 'PNG'
                bake_img.save()
            finally:
                try:
                    temp_nodes.remove(bake_node)
                except Exception:
                    pass
                assign_active_material(original_material)
                obj.active_material = original_material

            return os.path.isfile(outpath)
        
        baked_images_paths = {}
        active_channels = []
        for ch_name, b_type, suffix, is_noncolor in channels:
            socket = bsdf.inputs.get(ch_name)
            if socket and socket.is_linked:
                active_channels.append((ch_name, b_type, suffix, is_noncolor))

        total_progress_steps = float(max((len(active_channels) * 3) + 2, 1))
        completed_steps = 0.0
        _progress_set_label("Preparing bake workflow")
        _progress_begin(context, total_progress_steps)

        try:
            for ch_name, b_type, suffix, is_noncolor in active_channels:
                _progress_set_label(f"Preparing {ch_name} map")
                _progress_update(context, completed_steps + 0.2)
                socket = bsdf.inputs.get(ch_name)
                if not socket:
                    continue
                
                w, h = get_resolution(socket)
                
                needs_blur_val = 0.0
                needs_erode = bool(
                    s.show_optional_blur and suffix == 'Metalness' and s.erode_metalness_clusters and _HAS_NUMPY
                )
                if s.show_optional_blur:
                    if suffix == 'Roughness': needs_blur_val = s.blur_roughness
                    elif suffix == 'Metalness': needs_blur_val = s.blur_metalness
                
                ext_str = getattr(s, "export_format", "PNG")
                ext = '.png'
                if ext_str == 'JPEG': ext = '.jpg'
                elif ext_str == 'TIFF': ext = '.tif'
                outpath = os.path.join(processed_dir, f"{suffix}{ext}")

                if b_type == 'NORMAL':
                    if normal_direct_image_node and normal_direct_image_node.image:
                        _progress_set_label("Reusing normal map")
                        _export_image_copy(context, normal_direct_image_node.image, outpath)
                        if os.path.isfile(outpath):
                            baked_images_paths[suffix] = outpath
                            completed_steps += 2.0
                            _progress_update(context, completed_steps)
                            continue

                    _progress_set_label("Baking normal map")
                    if bake_normal_isolated(outpath, w, h, ext_str):
                        baked_images_paths[suffix] = outpath
                        completed_steps += 3.0
                        _progress_update(context, completed_steps)
                        continue

                    raise RuntimeError("Normal bake did not produce an output image.")

                is_direct, direct_node = is_direct_texture_connection(socket)
                
                # If blur is needed, force a real bake instead of copying the source image.
                if is_direct and direct_node and direct_node.image and (needs_blur_val <= 0.0) and not needs_erode:
                    _progress_set_label(f"Reusing {ch_name} map")
                    _export_image_copy(context, direct_node.image, outpath)
                    if os.path.isfile(outpath):
                        baked_images_paths[suffix] = outpath
                        completed_steps += 2.0
                        _progress_update(context, completed_steps)
                        continue

                img_name = f"{obj.name}_{suffix}_baked"
                bake_img = bpy.data.images.get(img_name)
                if bake_img:
                    bpy.data.images.remove(bake_img)
                
                bake_img = bpy.data.images.new(name=img_name, width=w, height=h, alpha=False, float_buffer=is_noncolor)
                _progress_set_label(f"Baking {ch_name}")
                _progress_update(context, completed_steps + 0.8)

                if is_noncolor:
                    try: bake_img.colorspace_settings.name = 'Non-Color'
                    except: pass

                bake_node = nodes.new('ShaderNodeTexImage')
                bake_node.image = bake_img
                
                if is_noncolor:
                    try: bake_node.image.colorspace_settings.name = 'Non-Color'
                    except: pass

                nodes.active = bake_node
                bake_node.select = True
                for n in nodes:
                    if n != bake_node:
                        n.select = False

                if b_type == 'EMIT':
                    if out_node.inputs['Surface'].is_linked:
                        for lk in list(out_node.inputs['Surface'].links):
                            links.remove(lk)
                    if emit_node.inputs['Color'].is_linked:
                        for lk in list(emit_node.inputs['Color'].links):
                            links.remove(lk)
                    src_socket = socket.links[0].from_socket
                    links.new(src_socket, emit_node.inputs['Color'])
                    links.new(emit_node.outputs['Emission'], out_node.inputs['Surface'])

                    bpy.ops.object.bake(type='EMIT', use_clear=True, margin=16)

                elif b_type == 'NORMAL':
                    if out_node.inputs['Surface'].is_linked:
                        for lk in list(out_node.inputs['Surface'].links):
                            links.remove(lk)
                    if orig_surface_link:
                        links.new(orig_surface_link, out_node.inputs['Surface'])

                    bpy.ops.object.bake(type='NORMAL', normal_space='TANGENT', use_clear=True, margin=16)
                
                if is_noncolor and needs_erode:
                    _progress_set_label(f"Eroding small clusters in {ch_name}")
                    w_e, h_e = bake_img.size[0], bake_img.size[1]
                    ebuf = np.empty(w_e * h_e * 4, dtype=np.float32)
                    if hasattr(bake_img.pixels, "foreach_get"): bake_img.pixels.foreach_get(ebuf)
                    else: ebuf[:] = np.array(bake_img.pixels[:], dtype=np.float32)
                    ergb = ebuf.reshape((h_e, w_e, 4))
                    side = (float(s.erode_cluster_threshold) / 100.0) * w_e
                    cleaned = _remove_small_clusters(ergb[:, :, 0].copy(), max(1, int(round(side * side))))
                    ergb[:, :, 0] = cleaned
                    ergb[:, :, 1] = cleaned
                    ergb[:, :, 2] = cleaned
                    if hasattr(bake_img.pixels, "foreach_set"): bake_img.pixels.foreach_set(ebuf)
                    else: bake_img.pixels = ebuf.tolist()
                    bake_img.update()

                # --- FAST GAUSSIAN BLUR APPROXIMATION OVER BAKED MAP ---
                if is_noncolor and _HAS_NUMPY and needs_blur_val > 0.0:
                    _progress_set_label(f"Blurring {ch_name}")
                    w_b, h_b = bake_img.size[0], bake_img.size[1]
                    buf = np.empty(w_b * h_b * 4, dtype=np.float32)
                    if hasattr(bake_img.pixels, "foreach_get"): bake_img.pixels.foreach_get(buf)
                    else: buf[:] = np.array(bake_img.pixels[:], dtype=np.float32)
                    
                    rgb = buf.reshape((h_b, w_b, 4))
                    r_rad = max(1, int((needs_blur_val / 100.0) * w_b))
                    
                    vc = rgb[:, :, 0].copy()
                    for _ in range(3):
                        pad = np.pad(vc, ((0,0), (r_rad, r_rad)), mode='edge')
                        cum = np.zeros((h_b, w_b + 2*r_rad + 1), dtype=np.float32)
                        cum[:, 1:] = np.cumsum(pad, axis=1)
                        vc = (cum[:, 2*r_rad+1:] - cum[:, :-2*r_rad-1]) / (2*r_rad + 1)
                        
                        pad = np.pad(vc, ((r_rad, r_rad), (0,0)), mode='edge')
                        cum = np.zeros((h_b + 2*r_rad + 1, w_b), dtype=np.float32)
                        cum[1:, :] = np.cumsum(pad, axis=0)
                        vc = (cum[2*r_rad+1:, :] - cum[:-2*r_rad-1, :]) / (2*r_rad + 1)
                        
                    rgb[:, :, 0] = vc
                    rgb[:, :, 1] = vc
                    rgb[:, :, 2] = vc
                    
                    out_buf = rgb.flatten()
                    if hasattr(bake_img.pixels, "foreach_set"): bake_img.pixels.foreach_set(out_buf)
                    else: bake_img.pixels = out_buf.tolist()
                    bake_img.update()
                # -------------------------------------------------------

                bake_img.filepath_raw = outpath
                bake_img.file_format = ext_str if ext_str in ['PNG', 'JPEG', 'TIFF'] else 'PNG'
                bake_img.save()
                
                baked_images_paths[suffix] = outpath
                nodes.remove(bake_node)
                completed_steps += 3.0
                _progress_update(context, completed_steps)
                
            if baked_images_paths:
                # 1. EARLY RESTORE & CLEANUP
                if orig_surface_link and out_node and out_node.inputs['Surface'].is_linked:
                    for lk in list(out_node.inputs['Surface'].links):
                        try: links.remove(lk)
                        except: pass
                    try: links.new(orig_surface_link, out_node.inputs['Surface'])
                    except: pass

                node_to_del = nodes.get("CP2PBR_Bake_Emission")
                if node_to_del:
                    nodes.remove(node_to_del)

                _progress_set_label("Applying baked textures")
                if not _apply_baked_maps_to_material(mat, baked_images_paths):
                    raise RuntimeError("Could not apply the baked maps.")
                rebuilt_baked_material = True

            _progress_set_label("Bake complete")
            _progress_update(context, total_progress_steps)

        except Exception as e:
            self.report({'ERROR'}, f"Bake failed: {str(e)}")
        finally:
            if normal_source_material and normal_source_material.users == 0:
                try:
                    bpy.data.materials.remove(normal_source_material)
                except Exception:
                    pass
            if not rebuilt_baked_material and orig_surface_link and out_node and out_node.inputs['Surface'].is_linked:
                for lk in list(out_node.inputs['Surface'].links):
                    try: links.remove(lk)
                    except: pass
                try: links.new(orig_surface_link, out_node.inputs['Surface'])
                except: pass

            node_to_del = nodes.get("CP2PBR_Bake_Emission")
            if node_to_del:
                nodes.remove(node_to_del)

            # Restore render parameters
            scn.render.engine = old_engine
            scn.cycles.device = old_device
            scn.view_settings.view_transform = old_view_transform
            if hasattr(scn.cycles, 'use_denoising'):
                scn.cycles.use_denoising = old_denoise
            scn.cycles.samples = old_samples
            _progress_end(context)

        self.report({'INFO'}, "Full Bake Complete and Material Applied!")
        return {'FINISHED'}

# ----------------------------
# UI Panels
# ----------------------------

class CP2PBR_OT_create_raw_shader_debug(Operator):
    bl_idname = "cp2pbr.create_raw_shader_debug"
    bl_label = "Create raw shader + debug maps"
    bl_options = {"REGISTER", "UNDO"}
    overwrite_targets: StringProperty(default="", options={'HIDDEN'})

    def draw(self, context):
        layout = self.layout
        layout.label(text="The following debug maps already exist and will be overwritten:")
        for name in [item for item in self.overwrite_targets.split("|") if item]:
            layout.label(text=name, icon='FILE_IMAGE')

    def invoke(self, context, event):
        _remember_progress_anchor(context, event)

        obj = _get_input_mesh(context)
        if not obj:
            return self.execute(context)

        s = context.scene.cp2pbr_settings
        reuse_ok = bool(s.reuse_existing_debug_maps and _debug_signature_matches(s))
        reused_existing_maps = bool(reuse_ok and _debug_maps_complete(s))
        if reused_existing_maps:
            return self.execute(context)

        targets = _debug_output_targets(s)
        if not targets:
            return self.execute(context)

        np_light_exists = os.path.isfile(_np_light_path(s))
        cp_light_exists = os.path.isfile(_cp_light_path(s))
        paths_to_check = []
        if not (reuse_ok and np_light_exists and cp_light_exists):
            paths_to_check.extend(targets[:2])
        paths_to_check.extend(targets[2:])

        existing_paths = _existing_output_paths(paths_to_check)
        if not existing_paths:
            return self.execute(context)

        self.overwrite_targets = "|".join(os.path.basename(path) for path in existing_paths)
        return context.window_manager.invoke_props_dialog(self, width=360)

    def modal(self, context, event):
        return _handle_deferred_modal(self, context, event, self._execute_sync)

    def execute(self, context):
        _remember_progress_anchor(context)
        return _start_deferred_task(self, context, "Preparing raw shader workflow")

    def _execute_sync(self, context):
        scn = context.scene
        s = scn.cp2pbr_settings
        obj = _require_input_mesh(context, self, "creating raw shader and debug maps")

        if not obj:
            return {'CANCELLED'}
        _sync_active_selected_object(context, obj)

        if scn.render.image_settings.file_format == 'FFMPEG':
            self.report(
                {'ERROR'},
                "Render Output is set to Video. In Output Properties, switch the media type/File Format to Image, then run the operator again."
            )
            return {'CANCELLED'}

        reuse_ok = bool(s.reuse_existing_debug_maps and _debug_signature_matches(s))
        reused_existing_maps = bool(reuse_ok and _debug_maps_complete(s))
        np_light_exists = os.path.isfile(_np_light_path(s))
        cp_light_exists = os.path.isfile(_cp_light_path(s))
        should_extract_lightness = (not reused_existing_maps) and (not (reuse_ok and np_light_exists and cp_light_exists))
        should_export_debug = not reused_existing_maps
        should_adjust = should_export_debug and any(_hist_modified(s, kind) for kind in _HIST_KINDS)
        total_steps = (
            1.0
            + (2.0 if should_extract_lightness else 0.0)
            + (1.0 if should_export_debug else 0.0)
            + (1.0 if should_adjust else 0.0)
        )
        completed_steps = 0.0

        weights = _normalized_rgb_weights(s)

        _progress_begin(context, total_steps)
        np_img = None
        cp_img = None
        try:
            if should_adjust:
                _progress_set_label("Applying histogram adjustments")
                _progress_update(context, completed_steps + 0.15)
                ok, msg = _export_adjusted_textures(context, s)
                if not ok:
                    self.report({"ERROR"}, msg)
                    return {"CANCELLED"}
                completed_steps += 1.0
                _progress_update(context, completed_steps)

            if should_extract_lightness:
                np_img = _load_image_raw(_source_texture_path(s, "np"))
                cp_img = _load_image_raw(_source_texture_path(s, "cp"))
                if np_img is None or cp_img is None:
                    self.report({"ERROR"}, "Could not load input textures.")
                    return {"CANCELLED"}

                _progress_set_label("Extracting NP light")
                _progress_update(context, completed_steps + 0.15)
                _extract_light_image_to_file(context, np_img, _np_light_path(s), ".tif", weights)
                completed_steps += 1.0
                _progress_update(context, completed_steps)

                _progress_set_label("Extracting CP light")
                _progress_update(context, completed_steps + 0.15)
                _extract_light_image_to_file(context, cp_img, _cp_light_path(s), ".tif", weights)
                completed_steps += 1.0
                _progress_update(context, completed_steps)

            if should_export_debug:
                _progress_set_label("Generating debug map")
                _progress_update(context, completed_steps + 0.15)
                ok, msg = _export_normalized_difference(
                    context,
                    out_name="Debug_Subtr_Normalized",
                    brightness_adj=s.cp_brightness_input,
                    show_progress=False,
                )
                if not ok:
                    self.report({"ERROR"}, msg)
                    return {"CANCELLED"}
                if s.export_rgb_diff_maps:
                    ok, msg = _export_rgb_difference_maps(context, s)
                    if not ok:
                        self.report({"ERROR"}, msg)
                        return {"CANCELLED"}
                completed_steps += 1.0
                _progress_update(context, completed_steps)

            _progress_set_label("Applying raw shader")
            _progress_update(context, completed_steps + 0.15)
            res = bpy.ops.cp2pbr.create_apply_shader()
            if "CANCELLED" in res:
                return {"CANCELLED"}
            if should_export_debug:
                _write_debug_signature(s)
            completed_steps += 1.0
            _progress_set_label("Raw shader workflow complete")
            _progress_update(context, total_steps)
        finally:
            _discard_image(np_img)
            _discard_image(cp_img)
            _progress_end(context)

        if reused_existing_maps:
            self.report({"INFO"}, "Existing debug maps found in the output folder. Raw shader applied without recalculating them.")
        else:
            self.report({"INFO"}, "Raw shader created + debug maps exported.")
        return {"FINISHED"}

def _reset_roughness_controls(settings):
    settings.roughness_black = 1.0
    settings.roughness_grey = 0.5
    settings.roughness_white = 0.0
    settings.roughness_black_col = (1.0, 1.0, 1.0, 1.0)
    settings.roughness_white_col = (0.0, 0.0, 0.0, 1.0)

def _reset_metalness_controls(settings):
    settings.metalness_black_col = (0.0, 0.0, 0.0, 1.0)
    settings.metalness_white_col = (1.0, 1.0, 1.0, 1.0)
    settings.metalness_black = 0.0
    settings.metalness_grey = 0.5
    settings.metalness_white = 1.0

def _reset_albedo_controls(settings):
    settings.metal_hsv_hue = 0.5
    settings.metal_hsv_saturation = 1.0
    settings.metal_hsv_value = 1.0

class CP2PBR_OT_reset_metalness_controls(Operator):
    bl_idname = "cp2pbr.reset_metalness_controls"
    bl_label = "Reset Metalness Controls"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        _reset_metalness_controls(context.scene.cp2pbr_settings)
        return {"FINISHED"}

class CP2PBR_OT_reset_roughness_controls(Operator):
    bl_idname = "cp2pbr.reset_roughness_controls"
    bl_label = "Reset Roughness Controls"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        _reset_roughness_controls(context.scene.cp2pbr_settings)
        return {"FINISHED"}

class CP2PBR_OT_reset_albedo_controls(Operator):
    bl_idname = "cp2pbr.reset_albedo_controls"
    bl_label = "Reset Albedo Controls"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        _reset_albedo_controls(context.scene.cp2pbr_settings)
        return {"FINISHED"}

class CP2PBR_OT_reset_hist_controls(Operator):
    bl_idname = "cp2pbr.reset_hist_controls"
    bl_label = "Reset Histogram Controls"
    bl_options = {"REGISTER", "UNDO"}

    kind: StringProperty(default="np", options={'HIDDEN'})

    def execute(self, context):
        s = context.scene.cp2pbr_settings
        setattr(s, f"{self.kind}_hist_black", 0.0)
        setattr(s, f"{self.kind}_hist_white", 1.0)
        setattr(s, f"{self.kind}_hist_black_col", (0.0, 0.0, 0.0, 1.0))
        setattr(s, f"{self.kind}_hist_white_col", (1.0, 1.0, 1.0, 1.0))
        return {"FINISHED"}

class CP2PBR_OT_save_calibration_txt(Operator):
    bl_idname = "cp2pbr.save_calibration_txt"
    bl_label = "Save calibration (txt)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.cp2pbr_settings
        ok, msg = _write_calibration_coeffs_file(s)
        if not ok:
            if msg == "no output folder is found":
                self.report({"WARNING"}, msg)
                return {"FINISHED"}
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        self.report({"INFO"}, f"Saved calibration: {os.path.basename(msg)}")
        return {"FINISHED"}

class CP2PBR_OT_load_calibration_txt(Operator, ImportHelper):
    bl_idname = "cp2pbr.load_calibration_txt"
    bl_label = "Load calibration (txt)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".txt"
    filter_glob: StringProperty(default="*.txt", options={'HIDDEN'})

    def execute(self, context):
        coeffs, msg = _load_calibration_coeffs_file(self.filepath)
        if not coeffs:
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        _apply_calibration_coeffs(context.scene.cp2pbr_settings, coeffs)
        self.report({"INFO"}, f"Loaded calibration: {os.path.basename(msg)}")
        return {"FINISHED"}

class CP2PBR_OT_save_hist_preset_txt(Operator):
    bl_idname = "cp2pbr.save_hist_preset_txt"
    bl_label = "Save raw shader + debug settings (txt)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        ok, msg = _write_hist_preset_file(context.scene.cp2pbr_settings)
        if not ok:
            if msg == "no output folder is found":
                self.report({"WARNING"}, msg)
                return {"FINISHED"}
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        self.report({"INFO"}, f"Saved raw shader + debug settings: {os.path.basename(msg)}")
        return {"FINISHED"}

class CP2PBR_OT_load_hist_preset_txt(Operator, ImportHelper):
    bl_idname = "cp2pbr.load_hist_preset_txt"
    bl_label = "Load raw shader + debug settings (txt)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".txt"
    filter_glob: StringProperty(default="*.txt", options={'HIDDEN'})

    def execute(self, context):
        coeffs, msg = _load_hist_preset_file(self.filepath)
        if not coeffs:
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        _apply_hist_preset(context.scene.cp2pbr_settings, coeffs)
        self.report({"INFO"}, f"Loaded raw shader + debug settings: {os.path.basename(msg)}")
        return {"FINISHED"}

class CP2PBR_OT_add_custom_normal_map(Operator):
    bl_idname = "cp2pbr.add_custom_normal_map"
    bl_label = "Add Custom Normal Map"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.cp2pbr_settings
        if settings.enable_fake_normal_map:
            self.report({"ERROR"}, "Disable Fake normal map before adding a custom normal map.")
            return {"CANCELLED"}

        obj = _require_input_mesh(context, self, "adding a custom normal map")
        if not obj:
            return {"CANCELLED"}

        mat = obj.active_material
        if not mat or not mat.use_nodes:
            self.report({"ERROR"}, "Input Mesh must have a valid node material.")
            return {"CANCELLED"}

        image_path = bpy.path.abspath(settings.custom_normal_map_path)
        if not image_path or not os.path.isfile(image_path):
            self.report({"ERROR"}, "Select a valid custom normal map path.")
            return {"CANCELLED"}

        normal_img = _load_image(image_path, force_reload=True)
        if not normal_img:
            self.report({"ERROR"}, "Could not load the selected normal map.")
            return {"CANCELLED"}
        _set_colorspace(normal_img, "Non-Color")

        node_tree = mat.node_tree
        nodes = node_tree.nodes
        links = node_tree.links

        bsdf = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
        if not bsdf:
            self.report({"ERROR"}, "Could not find Principled BSDF.")
            return {"CANCELLED"}

        normal_input = bsdf.inputs.get("Normal")
        if not normal_input:
            self.report({"ERROR"}, "Principled BSDF has no Normal input.")
            return {"CANCELLED"}

        tex_node = nodes.get(_CUSTOM_NORMAL_TEXTURE_NODE_NAME)
        if tex_node and tex_node.type != 'TEX_IMAGE':
            _remove_node_and_links(node_tree, tex_node)
            tex_node = None
        if not tex_node:
            tex_node = nodes.new("ShaderNodeTexImage")
            tex_node.name = tex_node.label = _CUSTOM_NORMAL_TEXTURE_NODE_NAME
            tex_node.location = (bsdf.location.x - 620, bsdf.location.y - 760)
        tex_node.image = normal_img
        _set_colorspace(tex_node.image, "Non-Color")

        normal_map_node = nodes.get(_CUSTOM_NORMAL_MAP_NODE_NAME)
        if normal_map_node and normal_map_node.type != 'NORMAL_MAP':
            _remove_node_and_links(node_tree, normal_map_node)
            normal_map_node = None
        if not normal_map_node:
            normal_map_node = nodes.new("ShaderNodeNormalMap")
            normal_map_node.name = normal_map_node.label = _CUSTOM_NORMAL_MAP_NODE_NAME
            normal_map_node.location = (bsdf.location.x - 300, bsdf.location.y - 760)

        for link in list(normal_map_node.inputs["Color"].links):
            links.remove(link)
        links.new(tex_node.outputs["Color"], normal_map_node.inputs["Color"])

        for link in list(normal_input.links):
            links.remove(link)
        links.new(normal_map_node.outputs["Normal"], normal_input)

        mat.pop(_NORMAL_LINK_BACKUP_PROP, None)
        self.report({"INFO"}, "Custom normal map added to the shader.")
        return {"FINISHED"}

class CP2PBR_PT_panel(Panel):
    bl_idname = "CP2PBR_PT_panel"
    bl_label = "Import and settings"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CP2PBR"
    
    def draw(self, context):
        s = context.scene.cp2pbr_settings
        layout = self.layout
        
        import_box = layout.box()
        import_box.label(text="Import", icon='IMAGE_DATA')
        import_box.prop(s, "input_mesh", text="Input Mesh", icon='EYEDROPPER')
        import_box.prop(s, "np_path", text="NCP")
        import_box.prop(s, "cp_path", text="CP")
        
        output_box = layout.box()
        output_box.label(text="Output", icon='FILE_FOLDER')
        output_box.prop(s, "output_dir", text="Folder")
        
        format_box = layout.box()
        format_box.label(text="Format and resolution", icon='SETTINGS')
        format_box.prop(s, "export_format", text="Format")
        format_box.prop(s, "enable_resize", text="Reduce Resolution")
        if s.enable_resize:
            format_box.prop(s, "target_width", text="Max Width")

class CP2PBR_PT_raw_panel(Panel):
    bl_idname = "CP2PBR_PT_raw_panel"
    bl_label = "Raw shader + debug maps"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CP2PBR"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.cp2pbr_settings
        raw_box = self.layout.box()
        raw_box.prop(s, "reuse_existing_debug_maps", text="Reuse existing debug maps")
        raw_box.prop(s, "enable_normalization", text="Normalization")
        if s.enable_normalization:
            raw_box.prop(s, "reuse_histogram_params_from_file", text="Load normalization preset (txt)")
            if s.reuse_histogram_params_from_file:
                raw_box.prop(s, "histogram_params_file", text="Parameters file")

        adv_row = raw_box.row(align=True)
        adv_row.prop(s, "show_advanced_debug", icon="TRIA_DOWN" if s.show_advanced_debug else "TRIA_RIGHT", text="Advanced", emboss=False)
        if s.show_advanced_debug:
            raw_box.label(text="RGB weights")
            weights_row = raw_box.row(align=True)
            weights_row.prop(s, "debug_weight_r", text="R")
            weights_row.prop(s, "debug_weight_g", text="G")
            weights_row.prop(s, "debug_weight_b", text="B")
            raw_box.prop(s, "export_rgb_diff_maps", text="Export RGB difference maps")
            raw_box.separator(factor=0.4)
            for kind, map_type in (("np", "NP_HIST"), ("cp", "CP_HIST")):
                hist_col = raw_box.column(align=True)
                title_row = hist_col.row(align=True)
                title_row.alert = True
                title_row.operator(
                    "cp2pbr.preview_map",
                    text=f"{_HIST_LABELS[kind]} histogram preview",
                    icon='HIDE_OFF',
                ).map_type = map_type
                reset_op = title_row.row(align=True)
                reset_op.operator("cp2pbr.reset_hist_controls", text="", icon='FILE_REFRESH').kind = kind

                min_row = hist_col.row(align=True)
                min_color = min_row.row(align=True); min_color.scale_x = 0.35
                min_color.prop(s, f"{kind}_hist_black_col", text="")
                min_row.prop(s, f"{kind}_hist_black", text=f"{_HIST_LABELS[kind]} Min")
                max_row = hist_col.row(align=True)
                max_color = max_row.row(align=True); max_color.scale_x = 0.35
                max_color.prop(s, f"{kind}_hist_white_col", text="")
                max_row.prop(s, f"{kind}_hist_white", text=f"{_HIST_LABELS[kind]} Max")
                raw_box.separator(factor=0.4)
        hist_io_row = raw_box.row(align=True)
        hist_io_row.operator("cp2pbr.save_hist_preset_txt", text="Save settings (txt)", icon='EXPORT')
        hist_io_row.operator("cp2pbr.load_hist_preset_txt", text="Load settings", icon='IMPORT')
        raw_box.separator(factor=0.4)
        raw_button_row = raw_box.row()
        raw_button_row.scale_y = 1.15
        raw_button_row.operator(
            "cp2pbr.create_raw_shader_debug",
            text="Create raw shader + debug maps",
            icon='SHADING_RENDERED',
            depress=True)
        raw_box.label(text="Debug maps are saved as 16-bit TIFF.")

class CP2PBR_PT_shader_panel(Panel):
    bl_idname = "CP2PBR_PT_shader_panel"
    bl_label = "Shader"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CP2PBR"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.cp2pbr_settings
        layout = self.layout

        calibration_box = layout.box()
        calibration_box.label(text="Shader calibration", icon='MATERIAL')

        mat_row = calibration_box.row(align=True)
        mat_row.scale_y = 1.15
        mat_row.operator("cp2pbr.preview_map", text="Material preview", icon='MATERIAL', depress=True).map_type = 'FINAL'

        calibration_box.separator(factor=0.8)
        calibration_box.prop(s, "clipping_preview", text="clipping preview")

        m_col = calibration_box.column(align=True)
        m_row = m_col.row(align=True)
        m_row.alert = True
        m_row.operator("cp2pbr.preview_map", text="Metalness preview", icon='HIDE_OFF').map_type = 'METALNESS'
        m_reset = m_row.row(align=True)
        m_reset.scale_x = 1.0
        m_reset.operator("cp2pbr.reset_metalness_controls", text="", icon='FILE_REFRESH')

        m1 = m_col.row(align=True)
        c1m = m1.row(align=True); c1m.scale_x = 0.35
        c1m.prop(s, "metalness_black_col", text="")
        m1.prop(s, "metalness_black", text="Metalness Min")
        m2 = m_col.row(align=True)
        c2m = m2.row(align=True); c2m.scale_x = 0.35
        c2m.label(text="")
        m2.prop(s, "metalness_grey", text="Metalness Blend")
        m3 = m_col.row(align=True)
        c3m = m3.row(align=True); c3m.scale_x = 0.35
        c3m.prop(s, "metalness_white_col", text="")
        m3.prop(s, "metalness_white", text="Metalness Max")

        calibration_box.separator(factor=0.6)
        r_col = calibration_box.column(align=True)
        r_row = r_col.row(align=True)
        r_row.alert = True
        r_row.operator("cp2pbr.preview_map", text="Roughness preview", icon='HIDE_OFF').map_type = 'ROUGHNESS'
        r_reset = r_row.row(align=True)
        r_reset.scale_x = 1.0
        r_reset.operator("cp2pbr.reset_roughness_controls", text="", icon='FILE_REFRESH')

        r3 = r_col.row(align=True)
        c3 = r3.row(align=True); c3.scale_x = 0.35
        c3.prop(s, "roughness_white_col", text="")
        r3.prop(s, "roughness_white", text="Roughness Min")

        r2 = r_col.row(align=True)
        c2 = r2.row(align=True); c2.scale_x = 0.35
        c2.label(text="")
        r2.prop(s, "roughness_grey", text="Roughness Blend")

        r1 = r_col.row(align=True)
        c1 = r1.row(align=True); c1.scale_x = 0.35
        c1.prop(s, "roughness_black_col", text="")
        r1.prop(s, "roughness_black", text="Roughness Max")

        calibration_box.separator(factor=0.6)
        a_col = calibration_box.column(align=True)
        albedo_row = a_col.row(align=True)
        albedo_row.alert = True
        albedo_row.operator("cp2pbr.preview_map", text="Albedo preview", icon='HIDE_OFF').map_type = 'ALBEDO'
        a_reset = albedo_row.row(align=True)
        a_reset.scale_x = 1.0
        a_reset.operator("cp2pbr.reset_albedo_controls", text="", icon='FILE_REFRESH')

        a_col.prop(s, "metal_hsv_hue", text="Hue (metal)")
        a_col.prop(s, "metal_hsv_saturation", text="Saturation (metal)")
        a_col.prop(s, "metal_hsv_value", text="Value (metal)")

        calibration_box.separator(factor=0.6)
        normal_col = calibration_box.column(align=True)
        normal_header = normal_col.row(align=True)
        normal_header.alert = True
        normal_header.operator("cp2pbr.preview_map", text="Normal Preview", icon='HIDE_OFF').map_type = 'NORMAL'

        if s.enable_fake_normal_map:
            normal_distance_row = normal_col.row(align=True)
            normal_distance_row.prop(s, "fake_normal_distance", text="Distance")
        else:
            normal_path_row = normal_col.row(align=True)
            normal_path_row.prop(s, "custom_normal_map_path", text="")
            normal_apply_row = normal_col.row(align=True)
            normal_apply_row.operator("cp2pbr.add_custom_normal_map", text="Assign custon Normal")
        normal_col.separator(factor=0.5)
        normal_toggle_row = normal_col.row(align=True)
        normal_toggle_row.prop(s, "enable_fake_normal_map", text="Fake normal map")

        calibration_box.separator(factor=0.8)
        calibration_io_row = calibration_box.row(align=True)
        calibration_io_row.operator("cp2pbr.save_calibration_txt", text="Save calibration (txt)", icon='EXPORT')
        calibration_io_row.operator("cp2pbr.load_calibration_txt", text="Load calibration parameters", icon='IMPORT')

class CP2PBR_PT_bake_panel(Panel):
    bl_idname = "CP2PBR_PT_bake_panel"
    bl_label = "Bake textures"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CP2PBR"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.cp2pbr_settings
        layout = self.layout
        bake_box = layout.box()
        bake_box.label(text="Bake textures", icon='NODE_TEXTURE')
        bake_button_row = bake_box.row()
        bake_button_row.scale_y = 1.25
        bake_button_row.operator("cp2pbr.full_bake", text="Bake Maps & Apply Shader", icon='RENDER_STILL', depress=True)
        bake_box.prop(s, "reuse_existing_baked_maps", text="Reuse previously baked maps if available")
        bake_box.separator(factor=0.4)
        blur_row = bake_box.row(align=True)
        blur_row.prop(s, "show_optional_blur", icon="TRIA_DOWN" if s.show_optional_blur else "TRIA_RIGHT", text="Advanced export option", emboss=False)

        if s.show_optional_blur:
            if not _HAS_NUMPY:
                bake_box.label(text="Numpy required for these options!", icon='ERROR')
            else:
                bake_box.prop(s, "blur_roughness", text="Roughness Blur %")
                bake_box.prop(s, "blur_metalness", text="Metalness Blur %")
                bake_box.prop(s, "erode_metalness_clusters", text="Remove small metallic islands")
                if s.erode_metalness_clusters:
                    bake_box.prop(s, "erode_cluster_threshold", text="Threshold (% of image width)")

# ----------------------------
# Registration
# ----------------------------

classes = (
    CP2PBR_Settings,
    CP2PBR_OT_extract_lightness,
    CP2PBR_OT_preview_map,
    CP2PBR_OT_reset_metalness_controls,
    CP2PBR_OT_reset_roughness_controls,
    CP2PBR_OT_reset_albedo_controls,
    CP2PBR_OT_reset_hist_controls,
    CP2PBR_OT_save_calibration_txt,
    CP2PBR_OT_load_calibration_txt,
    CP2PBR_OT_save_hist_preset_txt,
    CP2PBR_OT_load_hist_preset_txt,
    CP2PBR_OT_add_custom_normal_map,
    CP2PBR_OT_export_debug_map,
    CP2PBR_OT_create_apply_shader,
    CP2PBR_OT_create_raw_shader_debug,
    CP2PBR_OT_full_bake,
    CP2PBR_PT_panel,
    CP2PBR_PT_raw_panel,
    CP2PBR_PT_shader_panel,
    CP2PBR_PT_bake_panel,
)

def register_addon():
    for c in classes: bpy.utils.register_class(c)
    bpy.types.Scene.cp2pbr_settings = PointerProperty(type=CP2PBR_Settings)

def unregister_addon():
    _reset_progress_feedback()
    del bpy.types.Scene.cp2pbr_settings
    for c in reversed(classes): bpy.utils.unregister_class(c)
