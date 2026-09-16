import os
import csv
import json
from datetime import datetime

import cv2
import numpy as np
from track_stereo import CameraTracker

from stereo_config_USB import (
    APP_CONFIG,
    ORB_DEFAULTS,
    TRACKER_DEFAULTS,
    MODEL_ESTIMATION_DEFAULTS,
    EVALUATE_PAIR_CONFIG,
    POSE_MATCH_CONFIG,
    PNP_CONFIG,
    KEYFRAME_CONFIG,
    CONFIGS,
    DISPLAY_CONFIG,
    METRIC_SCALE_CONFIG,
    CROSS_CALIB,
    CROSS_PAIRS,
    CROSS_MATCH_CONFIG,
    GEOMETRY_CONFIG,
    RECORDING_CONFIG,
    LOGGING_CONFIG,
)


# Вычисление матриц поворотов вокруг основной
def build_camera_to_anchor_rotations(cross_calib, anchor = 1):
    graph = {}
    for (i, j), calib in cross_calib.items():
        R_ij = np.asarray(calib['R'], dtype = np.float64)
        graph.setdefault(i, []).append((j, R_ij))
        graph.setdefault(j, []).append((i, R_ij.T))
    rotations = {anchor: np.eye(3, dtype = np.float64)}
    queue = [anchor]
    while queue:
        cur = queue.pop(0)
        R_cur_to_anchor = rotations[cur]
        for nxt, R_cur_to_nxt in graph.get(cur, []):
            if nxt in rotations:
                continue
            rotations[nxt] = R_cur_to_anchor @ R_cur_to_nxt.T
            queue.append(nxt)
            
    return rotations


# Поворот камер относительно основной
def vector_to_anchor_axes(vec, cam_id, rotations, placement = None):
    arr = np.asarray(vec, dtype = np.float64).reshape(3)
    if cam_id in rotations:
        return rotations[cam_id] @ arr
    if placement is not None:
        return vector_to_front_axes(arr, placement)
    
    return arr


# Добавление данный с зон стереопересечений
def append_stereo_observation(observations, cam_id, pixels, pts3d_metric, descriptors, pair_name):
    if len(pixels) == 0 or len(pts3d_metric) == 0 or descriptors is None or len(descriptors)  == 0:
        return

    pixels = np.asarray(pixels, dtype = np.float64).reshape(-1, 2)
    pts3d_metric = np.asarray(pts3d_metric, dtype = np.float64).reshape(-1, 3)
    descriptors = np.asarray(descriptors, dtype = np.uint8).reshape(-1, 32)

    n = min(len(pixels), len(pts3d_metric), len(descriptors))
    if n <= 0:
        return

    observations.setdefault(cam_id, {
        'pixels': [],
        'points3d': [],
        'descriptors': [],
        'pair_names': [],
        'sources': []
    })
    observations[cam_id]['pixels'].append(pixels[:n])
    observations[cam_id]['points3d'].append(pts3d_metric[:n])
    observations[cam_id]['descriptors'].append(descriptors[:n])
    observations[cam_id]['pair_names'].append(np.full(n, pair_name, dtype = object))
    observations[cam_id]['sources'].append(pair_name)


# Объединение стереоданных в массив
def merge_stereo_observation(obs):
    if obs is None or not obs.get('pixels'):
        return None

    return {
        'pixels': np.vstack(obs['pixels']),
        'points3d': np.vstack(obs['points3d']),
        'descriptors': np.vstack(obs['descriptors']),
        'pair_names': np.concatenate(obs['pair_names']),
        'source': '+'.join(obs.get('sources', []))
    }


# Сопоставление стерео-точек
def associate_by_grid_hamming(unit_pixels, unit_descriptors,
                              stereo_pixels, stereo_descriptors,
                              search_radius_px, hamming_threshold):
    unit_pixels = np.asarray(unit_pixels, dtype = np.float64).reshape(-1, 2)
    stereo_pixels = np.asarray(stereo_pixels, dtype = np.float64).reshape(-1, 2)
    unit_descriptors = np.asarray(unit_descriptors, dtype = np.uint8).reshape(-1, 32)
    stereo_descriptors = np.asarray(stereo_descriptors, dtype = np.uint8).reshape(-1, 32)

    n_unit = min(len(unit_pixels), len(unit_descriptors))
    n_stereo = min(len(stereo_pixels), len(stereo_descriptors))
    unit_pixels = unit_pixels[:n_unit]
    unit_descriptors = unit_descriptors[:n_unit]
    stereo_pixels = stereo_pixels[:n_stereo]
    stereo_descriptors = stereo_descriptors[:n_stereo]

    if n_unit  == 0 or n_stereo  == 0:
        return (np.empty(0, dtype = np.int32),
                np.empty(0, dtype = np.int32),
                np.empty(0, dtype = np.float64))

    search_radius_px = float(search_radius_px)
    hamming_threshold = float(hamming_threshold)
    cell_size = max(4.0, search_radius_px)

    # Создание сетки
    max_u = max(float(np.max(unit_pixels[:, 0])), float(np.max(stereo_pixels[:, 0])), 1.0)
    max_v = max(float(np.max(unit_pixels[:, 1])), float(np.max(stereo_pixels[:, 1])), 1.0)
    grid_cols = max(1, int(np.ceil((max_u + 1.0) / cell_size)))
    grid_rows = max(1, int(np.ceil((max_v + 1.0) / cell_size)))

    grid = [[[] for _ in range(grid_cols)] for _ in range(grid_rows)]
    for si, (u, v) in enumerate(stereo_pixels):
        cx = int(u // cell_size)
        cy = int(v // cell_size)
        if 0 <= cx < grid_cols and 0 <= cy < grid_rows:
            grid[cy][cx].append(si)

    cell_span = int(np.ceil(search_radius_px / cell_size))
    candidates_global = []

    # Поиск соответствий
    for ui, (u, v) in enumerate(unit_pixels):
        cx = int(u // cell_size)
        cy = int(v // cell_size)

        for dy in range(-cell_span, cell_span + 1):
            for dx in range(-cell_span, cell_span + 1):
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < grid_rows and 0 <= nx < grid_cols):
                    continue

                for si in grid[ny][nx]:
                    pixel_dist = float(np.linalg.norm(unit_pixels[ui] - stereo_pixels[si]))
                    if pixel_dist > search_radius_px:
                        continue

                    hamming = float(cv2.norm(unit_descriptors[ui], stereo_descriptors[si], cv2.NORM_HAMMING))
                    if hamming <= hamming_threshold:
                        candidates_global.append((hamming, pixel_dist, ui, si))

    # Сортировка по Хеммингу и расстоянию
    candidates_global.sort(key = lambda x: (x[0], x[1]))

    # Выбор уникальных соответствий
    used_u, used_s = set(), set()
    u_idx, s_idx, hamming_values = [], [], []
    for hamming, _, ui, si in candidates_global:
        if ui in used_u or si in used_s:
            continue
        used_u.add(ui)
        used_s.add(si)
        u_idx.append(ui)
        s_idx.append(si)
        hamming_values.append(hamming)

    return (np.asarray(u_idx, dtype = np.int32),
            np.asarray(s_idx, dtype = np.int32),
            np.asarray(hamming_values, dtype = np.float64))


# Вычисление коэффициента масштаба межу реальными и условными координатами
def estimate_metric_scale_simple(tracker, stereo_pixels, stereo_points3d_metric,
                                 stereo_descriptors, pair_names, cfg):
    unit_pixels, unit_points, unit_descriptors = tracker.get_current_unit_landmarks_with_descriptors()

    stereo_pixels = np.asarray(stereo_pixels, dtype = np.float64).reshape(-1, 2)
    stereo_points3d_metric = np.asarray(stereo_points3d_metric, dtype = np.float64).reshape(-1, 3)
    stereo_descriptors = np.asarray(stereo_descriptors, dtype = np.uint8).reshape(-1, 32)
    pair_names = np.asarray(pair_names, dtype = object)

    n_stereo = min(len(stereo_pixels), len(stereo_points3d_metric), len(stereo_descriptors), len(pair_names))
    stereo_pixels = stereo_pixels[:n_stereo]
    stereo_points3d_metric = stereo_points3d_metric[:n_stereo]
    stereo_descriptors = stereo_descriptors[:n_stereo]
    pair_names = pair_names[:n_stereo]

    # Сопоставление точек
    u_idx, s_idx, hamming_values = associate_by_grid_hamming(
        unit_pixels, unit_descriptors,
        stereo_pixels, stereo_descriptors,
        cfg.get('grid_search_radius_px', 10.0),
        cfg.get('hamming_threshold', 200)
    )
    if len(u_idx) < cfg.get('min_scale_points', 6):
        return None

    Pu = unit_points[u_idx]
    Pm = stereo_points3d_metric[s_idx]
    unit_pix_sel = unit_pixels[u_idx]
    stereo_pix_sel = stereo_pixels[s_idx]
    pair_sel = pair_names[s_idx]

    # Фильтрация по глубине
    finite = np.isfinite(Pu).all(axis = 1) & np.isfinite(Pm).all(axis = 1)
    finite &= np.isfinite(unit_pix_sel).all(axis = 1) & np.isfinite(stereo_pix_sel).all(axis = 1)
    finite &= (Pu[:, 2] > cfg.get('min_depth', 1e-6)) & (Pm[:, 2] > cfg.get('min_depth', 1e-6))

    Pu = Pu[finite]
    Pm = Pm[finite]
    unit_pix_sel = unit_pix_sel[finite]
    stereo_pix_sel = stereo_pix_sel[finite]
    pair_sel = pair_sel[finite]
    hamming_values = hamming_values[finite]

    if len(Pu) < cfg.get('min_scale_points', 6):
        return None

    cam = tracker.camera

    # Проверка ошибок репроекции
    reproj_unit = cam.ray_to_pixel_vectorized(cam.point_to_ray_vectorized(Pu))
    reproj_metric = cam.ray_to_pixel_vectorized(cam.point_to_ray_vectorized(Pm))

    err_unit = np.linalg.norm(reproj_unit - stereo_pix_sel, axis = 1)
    err_metric = np.linalg.norm(reproj_metric - unit_pix_sel, axis = 1)
    reproj_err = np.maximum(err_unit, err_metric)
    reproj_ok = np.isfinite(reproj_err) & (reproj_err <= cfg.get('scale_reproj_error_px', 5.0))

    Pu = Pu[reproj_ok]
    Pm = Pm[reproj_ok]
    unit_pix_sel = unit_pix_sel[reproj_ok]
    stereo_pix_sel = stereo_pix_sel[reproj_ok]
    pair_sel = pair_sel[reproj_ok]
    hamming_values = hamming_values[reproj_ok]
    reproj_err = reproj_err[reproj_ok]

    if len(Pu) < cfg.get('min_scale_points', 6):
        return None

    # Вычисление масштаба для каждой точки
    norm_u = np.linalg.norm(Pu, axis = 1)
    norm_m = np.linalg.norm(Pm, axis = 1)
    valid_scale = np.isfinite(norm_u) & np.isfinite(norm_m) & (norm_u > 1e-12) & (norm_m > 1e-12)

    scales = norm_m[valid_scale] / norm_u[valid_scale]
    Pu_scale = Pu[valid_scale]
    Pm_scale = Pm[valid_scale]
    unit_pix_scale = unit_pix_sel[valid_scale]
    stereo_pix_scale = stereo_pix_sel[valid_scale]
    pair_sel = pair_sel[valid_scale]
    hamming_values = hamming_values[valid_scale]
    reproj_err = reproj_err[valid_scale]

    valid_scale_values = np.isfinite(scales) & (scales > 0)
    scales = scales[valid_scale_values]
    Pu_scale = Pu_scale[valid_scale_values]
    Pm_scale = Pm_scale[valid_scale_values]
    unit_pix_scale = unit_pix_scale[valid_scale_values]
    stereo_pix_scale = stereo_pix_scale[valid_scale_values]
    pair_sel = pair_sel[valid_scale_values]
    hamming_values = hamming_values[valid_scale_values]
    reproj_err = reproj_err[valid_scale_values]

    if len(scales) < cfg.get('min_scale_points', 6):
        return None

    # Статистика по каждой стереопаре
    pair_means = {}
    for pair_name in np.unique(pair_sel):
        mask = pair_sel  == pair_name
        pair_means[str(pair_name)] = {
            'mean_scale': float(np.mean(scales[mask])),
            'points': int(np.sum(mask)),
        }

    return {
        'scale': float(np.mean(scales)),
        'points': int(len(scales)),
        'pair_means': pair_means,
        'mean_hamming': float(np.mean(hamming_values)) if len(hamming_values) else np.nan,
        'mean_reproj_error': float(np.mean(reproj_err)) if len(reproj_err) else np.nan,
        'point_details': {
            'unit_points': Pu_scale.astype(np.float64, copy = True),
            'metric_points': Pm_scale.astype(np.float64, copy = True),
            'unit_pixels': unit_pix_scale.astype(np.float64, copy = True),
            'stereo_pixels': stereo_pix_scale.astype(np.float64, copy = True),
            'pair_names': pair_sel.astype(object, copy = True),
            'scales': scales.astype(np.float64, copy = True),
            'hamming': hamming_values.astype(np.float64, copy = True),
            'reproj_error': reproj_err.astype(np.float64, copy = True),
        }
    }


# Соединение изображений с камер
def pad_to_max_size(images, target_w = None, target_h = None, pad_color = (255, 255, 255)):
    if not images:
        return images
    if target_w is None:
        target_w = max(img.shape[1] for img in images)
    if target_h is None:
        target_h = max(img.shape[0] for img in images)
    padded = []
    
    for img in images:
        h, w = img.shape[:2]
        top = (target_h - h) // 2
        bottom = target_h - h - top
        left = (target_w - w) // 2
        right = target_w - w - left
        padded_img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value = pad_color)
        padded.append(padded_img)
        
    return padded


# Заглушка при отсутствии изображения
def create_dummy_canvas(frame_shape, cam_id, panel_width, message, font_scale, thickness):
    h, w = frame_shape[:2]
    canvas = np.ones((h, w + panel_width, 3), dtype = np.uint8) * 255
    cv2.putText(canvas, f"Cam {cam_id}: {message}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
    
    return canvas


# Вставка кадра в изображение
def paste_video_layer_into_canvas(canvas, frame_layer):
    if canvas is None or frame_layer is None:
        return canvas
    
    h = min(canvas.shape[0], frame_layer.shape[0])
    w = min(frame_layer.shape[1], canvas.shape[1])
    x0 = canvas.shape[1] - frame_layer.shape[1]
    
    if x0 < 0:
        src_x0 = -x0
        canvas[:h, :w] = frame_layer[:h, src_x0:src_x0 + w]
        return canvas
    canvas[:h, x0:x0 + w] = frame_layer[:h, :w]
    
    return canvas


# Визуализация точек найденных на соседних камерах (квадраты разных цветов) 
def draw_cross_camera_match(frame, point, color, text = None):
    if frame is None:
        return
    h, w = frame.shape[:2]
    u, v = int(round(point[0])), int(round(point[1]))
    if not (0 <= u < w and 0 <= v < h):
        return
    cv2.rectangle(frame, (u - 3, v - 3), (u + 3, v + 3), color, 1)
    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        text_color = (0, 0, 0)
        tx = min(max(u + 5, 0), max(w - 170, 0))
        ty = min(max(v - 5, 12), h - 5)
        cv2.putText(frame, text, (tx, ty), font, font_scale, text_color, thickness)


# Добавление точки с соседних камер
def append_cross_camera_visual_observation(buffer, cam_id, point, color, text = None):
    if point is None:
        return
    p = np.asarray(point, dtype = np.float64).reshape(2)
    if not np.isfinite(p).all():
        return
    buffer.setdefault(cam_id, []).append({
        'point': p,
        'color': tuple(int(c) for c in color),
        'text': text,
    })


# Визуализация точек найденных на соседних камерах без повторений
def draw_cross_camera_observations_dedup(frame, observations, radius_px = 5.0):
    if frame is None or not observations:
        return 0
    
    h, w = frame.shape[:2]
    radius2 = float(radius_px) * float(radius_px)
    accepted = []
    drawn = 0
    
    for obs in observations:
        p = np.asarray(obs.get('point'), dtype = np.float64).reshape(2)
        if not np.isfinite(p).all():
            continue
        u, v = int(round(p[0])), int(round(p[1]))
        if not (0 <= u < w and 0 <= v < h):
            continue
        duplicate = False
        for q in accepted:
            if float(np.sum((p - q) ** 2)) <= radius2:
                duplicate = True
                break
        if duplicate:
            continue
        accepted.append(p)
        draw_cross_camera_match(frame, p, obs.get('color', (255, 0, 0)), obs.get('text'))
        drawn += 1
        
    return drawn


VALID_PLACEMENTS = {'front', 'back', 'right', 'left'}
PLACEMENT_AXIS_MAP = {
    'front': (0, 1, 2),
    'back': (0, 1, 2),
    'right': (2, 1, 0),
    'left': (2, 1, 0),
}


# Нормализация строки
def normalize_placement(value):
    placement = str(value).strip().lower()
    if placement not in VALID_PLACEMENTS:
        allowed = ', '.join(sorted(VALID_PLACEMENTS))
        raise ValueError(f"Unknown camera placement '{value}'. Use one of: {allowed}")

    return placement


# Приведение осей к общим
def vector_to_front_axes(vec, placement):
    placement = normalize_placement(placement)
    idx = PLACEMENT_AXIS_MAP.get(placement, PLACEMENT_AXIS_MAP['front'])
    arr = np.asarray(vec, dtype = np.float64).reshape(3)
    return arr[list(idx)]


# Среднее для углов
def circular_average_360(samples):
    if not samples:
        return np.zeros(3), np.zeros(3, dtype = np.int32)
    values = np.array([s['values'] for s in samples], dtype = np.float64)
    avg = np.zeros(3, dtype = np.float64)
    counts = np.zeros(3, dtype = np.int32)
    
    for axis in range(3):
        axis_values = values[:, axis]
        finite = np.isfinite(axis_values)
        if not np.any(finite):
            continue
        vals = np.mod(axis_values[finite], 360.0)
        counts[axis] = len(vals)
        radians = np.deg2rad(vals)
        mean_sin = float(np.mean(np.sin(radians)))
        mean_cos = float(np.mean(np.cos(radians)))
        if np.hypot(mean_sin, mean_cos) < 1e-12:
            avg[axis] = 0.0
        else:
            avg[axis] = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0
            
    return avg, counts


# Сбор состояний со всех активных трекеров
def collect_motion_states(trackers, active):
    states = []
    for i, tr in enumerate(trackers):
        if not active[i] or tr is None or tr.state != 'TRACKING':
            continue
        motion = tr.get_motion_state()
        if motion is not None:
            states.append(motion)
            
    return states


# Усреднение векторов через медиану
def robust_vector_average(samples):
    if not samples:
        return np.full(3, np.nan, dtype = np.float64), np.zeros(3, dtype = np.int32)
    values = np.array([s['values'] for s in samples], dtype = np.float64)
    avg = np.full(3, np.nan, dtype = np.float64)
    counts = np.zeros(3, dtype = np.int32)
    for axis in range(3):
        vals = values[:, axis]
        finite = np.isfinite(vals)
        counts[axis] = int(np.sum(finite))
        if counts[axis] > 0:
            avg[axis] = float(np.mean(vals[finite]))
            
    return avg, counts


# Усреднение движений со всех камер
def average_motion_by_placement(motion_states, rotations = None):
    result = {}
    translation_metrics = [
        'local_t_unit', 'global_t_unit',
        'local_t_metric', 'global_t_metric'
    ]
    anchor_metric_map = {
        'local_t_unit': 'local_t_anchor_unit',
        'global_t_unit': 'global_t_anchor_unit',
        'local_t_metric': 'local_t_anchor_metric',
        'global_t_metric': 'global_t_anchor_metric',
    }
    rotation_metrics = ['local_rot', 'global_rot']

    for metric in translation_metrics:
        samples = []
        anchor_metric = anchor_metric_map.get(metric)
        for motion in motion_states:
            placement = normalize_placement(motion.get('placement', 'front'))
            cam_id = motion.get('cam_id')

            values = motion.get(anchor_metric) if anchor_metric else None
            already_anchor = values is not None
            if values is None:
                values = motion.get(metric)
            if values is None:
                continue

            arr = np.asarray(values, dtype = np.float64).reshape(3)
            if not np.isfinite(arr).any():
                continue

            samples.append({
                'placement': placement,
                'values': arr if already_anchor else vector_to_anchor_axes(arr, cam_id, rotations or {}, placement),
                'cam_id': cam_id
            })
        avg, counts = robust_vector_average(samples)
        result[metric] = {'avg': avg, 'counts': counts}

    for metric in rotation_metrics:
        samples = []
        for motion in motion_states:
            values = motion.get(metric)
            if values is None:
                continue
            placement = normalize_placement(motion.get('placement', 'front'))
            samples.append({
                'placement': placement,
                'values': np.asarray(values, dtype = np.float64).reshape(3),
                'cam_id': motion.get('cam_id')
            })
        if metric  == 'global_rot':
            avg, counts = circular_average_360(samples)
        else:
            avg, counts = robust_vector_average(samples)
        result[metric] = {'avg': avg, 'counts': counts}

    return result


# Визуализация общей панели
def render_summary_column(avg_data, motion_states, trackers, active, height, width = 220, match_counts = None,
                          font_scale = None, thickness = None, line_height = None,
                          section_gap = None, margin_x = None, start_y = None):
    canvas = np.ones((height, width, 3), dtype = np.uint8) * 255
    font_scale = DISPLAY_CONFIG.get('font_scale', 0.45) if font_scale is None else font_scale
    thickness = DISPLAY_CONFIG.get('font_thickness', 1) if thickness is None else thickness
    line_height = DISPLAY_CONFIG.get('line_height', 18) if line_height is None else line_height
    section_gap = DISPLAY_CONFIG.get('section_gap', 6) if section_gap is None else section_gap
    margin_x = DISPLAY_CONFIG.get('text_margin_x', 10) if margin_x is None else margin_x
    y = DISPLAY_CONFIG.get('text_y_start', 22) if start_y is None else start_y

    def put(text):
        nonlocal y
        if y < height - 8:
            cv2.putText(canvas, str(text), (int(margin_x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        float(font_scale), (0, 0, 0), int(thickness))
        y += int(line_height)

    def gap():
        nonlocal y
        y += int(section_gap)

    if motion_states and avg_data is not None:
        def fmt(v, digits = 3):
            return "   -" if not np.isfinite(v) else f"{v: .{int(digits)}f}"

        put("Unit local")
        for label, val in [
            ('SULX', avg_data['local_t_unit']['avg'][0]),
            ('SULY', avg_data['local_t_unit']['avg'][1]),
            ('SULZ', avg_data['local_t_unit']['avg'][2]),
        ]:
            put(f"{label}: {fmt(val)}")
        gap()
        put("Unit global")
        for label, val in [
            ('SUGX', avg_data['global_t_unit']['avg'][0]),
            ('SUGY', avg_data['global_t_unit']['avg'][1]),
            ('SUGZ', avg_data['global_t_unit']['avg'][2]),
        ]:
            put(f"{label}: {fmt(val, 2)}")
        gap()
        put("Metric local, m")
        for label, val in [
            ('SMLX', avg_data['local_t_metric']['avg'][0]),
            ('SMLY', avg_data['local_t_metric']['avg'][1]),
            ('SMLZ', avg_data['local_t_metric']['avg'][2]),
        ]:
            put(f"{label}: {fmt(val)}")
        gap()
        put("Metric global, m")
        for label, val in [
            ('SMGX', avg_data['global_t_metric']['avg'][0]),
            ('SMGY', avg_data['global_t_metric']['avg'][1]),
            ('SMGZ', avg_data['global_t_metric']['avg'][2]),
        ]:
            put(f"{label}: {fmt(val, 2)}")
        gap()
        put("Rot deg")
        for label, val in [
            ('SRLX', avg_data['local_rot']['avg'][0]),
            ('SRLY', avg_data['local_rot']['avg'][1]),
            ('SRLZ', avg_data['local_rot']['avg'][2]),
        ]:
            put(f"{label}: {fmt(val)}")
        for label, val in [
            ('SGRX', avg_data['global_rot']['avg'][0]),
            ('SGRY', avg_data['global_rot']['avg'][1]),
            ('SGRZ', avg_data['global_rot']['avg'][2]),
        ]:
            put(f"{label}: {fmt(val, 1)}")

        gap()
        put("Scale:")
        for motion in motion_states:
            cam_id = motion.get('cam_id')
            if motion.get('metric_ready'):
                put(f"C{cam_id}: {motion.get('metric_scale', np.nan):.5f}")
            else:
                put(f"C{cam_id}: no")

        if match_counts:
            gap()
            put("Cross matches:")
            for name, cnt in match_counts.items():
                put(f"{name}: {cnt}")
                
        return canvas

    for i, tr in enumerate(trackers):
        if not active[i] or tr is None:
            continue
        placement = normalize_placement(getattr(tr, 'placement', 'front'))
        metrics = getattr(tr, 'last_init_metrics', None) or {}
        p = metrics.get('parallax_deg', None)
        sc = metrics.get('score', None)
        inn = metrics.get('inlier_count', None)
        p_text = '-' if p is None else f"{p:.2f}"
        sc_text = '-' if sc is None else f"{sc:.0f}"
        i_text = '-' if inn is None else f"{int(inn)}"
        put(f"C{i} {placement}")
        put(f"P {p_text}/{tr.min_parallax_deg:.1f}")
        put(f"SC {sc_text}/{tr.score_threshold:.0f}")
        put(f"I {i_text}/{tr.min_inliers_init}")
        gap()
    if match_counts:
        gap()
        put("Cross matches:")
        for name, cnt in match_counts.items():
            put(f"{name}: {cnt}")
            
    return canvas


# Проверка готовности всех камер к инициализации
def maybe_start_synchronous_initialization(trackers, active, video_canvases, kf_canvases, default_frame_size):
    init_indices = [
        i for i, tr in enumerate(trackers)
        if active[i] and tr is not None and tr.state  == 'INIT'
    ]
    if not init_indices:
        return False
    all_ready = all(trackers[i].is_ready_for_global_init() for i in init_indices)
    if not all_ready:
        return False
    print("\nВсе активные камеры готовы: выполняется синхронная инициализация")
    
    for i in init_indices:
        tr = trackers[i]
        video_canvas, kf_canvas, status = tr.complete_initialization_from_pending()
        if video_canvas is None:
            video_canvas = create_dummy_canvas(
                default_frame_size, i, DISPLAY_CONFIG['video_panel_width'], status,
                DISPLAY_CONFIG['camera_label_font_scale'], DISPLAY_CONFIG['font_thickness'])
        if kf_canvas is None:
            kf_canvas = create_dummy_canvas(
                default_frame_size, i, DISPLAY_CONFIG['kf_panel_width'], "no KF",
                DISPLAY_CONFIG['camera_label_font_scale'], DISPLAY_CONFIG['font_thickness'])
        video_canvases[i] = video_canvas.astype(np.uint8)
        kf_canvases[i] = kf_canvas.astype(np.uint8)
        
    return True


# Геометрическая фильтрация для точек на соседних камерах
def essential_filter(pts1_norm, pts2_norm, R, t, threshold = 0.01):
    tx = np.array([[0, -t[2], t[1]],
                   [t[2], 0, -t[0]],
                   [-t[1], t[0], 0]], dtype = np.float64)
    E = tx @ R
    pts1_h = np.hstack([pts1_norm, np.ones((len(pts1_norm), 1))])
    pts2_h = np.hstack([pts2_norm, np.ones((len(pts2_norm), 1))])
    errors = np.abs(np.sum(pts2_h * (pts1_h @ E.T), axis = 1))
    
    return errors < threshold


# Триангуляция точек с соседних камер
def triangulate_and_check_depth(pts1_rays, pts2_rays, R, t):
    n = len(pts1_rays)
    inliers = np.zeros(n, dtype = bool)
    points3d = np.full((n, 3), np.nan, dtype = np.float64)
    
    for i in range(n):
        ray1 = pts1_rays[i]
        ray2 = pts2_rays[i]
        if abs(ray1[2]) > 1e-8:
            u1 = ray1[0] / ray1[2]
            v1 = ray1[1] / ray1[2]
        else:
            u1 = ray1[0] * 1e6
            v1 = ray1[1] * 1e6
        if abs(ray2[2]) > 1e-8:
            u2 = ray2[0] / ray2[2]
            v2 = ray2[1] / ray2[2]
        else:
            u2 = ray2[0] * 1e6
            v2 = ray2[1] * 1e6
            
        P1 = np.hstack((np.eye(3), np.zeros((3,1))))
        P2 = np.hstack((R, t.reshape(3,1)))
        pt1_hom = np.array([u1, v1, 1.0])
        pt2_hom = np.array([u2, v2, 1.0])
        pts4d = cv2.triangulatePoints(P1, P2, pt1_hom[:2], pt2_hom[:2])
        pts3d = pts4d[:3] / (pts4d[3] + 1e-12)
        pts3d = pts3d.flatten()
        
        depth1 = pts3d[2]
        depth2 = (R @ pts3d + t)[2]
        if depth1 > 0 and depth2 > 0:
            inliers[i] = True
            points3d[i] = pts3d
            
    return inliers, points3d


# Настройка FPS для записи
def resolve_recording_fps(trackers, recording_config):
    configured_fps = recording_config.get('fps', None)
    if configured_fps is not None:
        try:
            fps = float(configured_fps)
            if np.isfinite(fps) and fps > 0:
                return fps
        except Exception:
            pass

    for tr in trackers:
        if tr is None or not hasattr(tr, 'cap'):
            continue
        fps = float(tr.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if np.isfinite(fps) and fps > 0:
            return fps
        
    return float(recording_config.get('fallback_fps', 30.0))


# Преобразование кадров к BGR
def ensure_bgr_uint8(frame):
    if frame is None:
        return None
    out = np.asarray(frame)
    if out.ndim  == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif out.ndim  == 3 and out.shape[2]  == 4:
        out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if not out.flags['C_CONTIGUOUS']:
        out = np.ascontiguousarray(out)
        
    return out

# Запись видео с отложенным открытием
class LazyVideoRecorder:
    def __init__(self, enabled, path, fps, fourcc = 'XVID', resize_on_size_change = True,
                 print_status = True, label = 'video'):
        self.enabled = bool(enabled)
        self.path = str(path)
        self.fps = float(fps)
        self.fourcc = str(fourcc or 'XVID')[:4].ljust(4)
        self.resize_on_size_change = bool(resize_on_size_change)
        self.print_status = bool(print_status)
        self.label = label
        self.writer = None
        self.frame_size = None
        self.frames_written = 0

    # Открытие видео с размерами первого кадра
    def _open(self, frame):
        h, w = frame.shape[:2]
        self.frame_size = (int(w), int(h))
        out_dir = os.path.dirname(self.path)
        if out_dir:
            os.makedirs(out_dir, exist_ok = True)
        fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
        self.writer = cv2.VideoWriter(self.path, fourcc_code, self.fps, self.frame_size)
        if not self.writer.isOpened():
            raise IOError(f"Не удалось открыть VideoWriter для {self.label}: {self.path}")
        if self.print_status:
            print(f"Запись {self.label}: {self.path}, {self.frame_size[0]}x{self.frame_size[1]} @ {self.fps:.3g} fps")

    # Запись кадра в видео
    def write(self, frame):
        if not self.enabled:
            return
        frame = ensure_bgr_uint8(frame)
        if frame is None:
            return
        if self.writer is None:
            self._open(frame)
        h, w = frame.shape[:2]
        if (w, h) != self.frame_size:
            if not self.resize_on_size_change:
                if self.print_status:
                    print(
                        f"Пропуск кадра {self.label}: размер {w}x{h} != "
                        f"{self.frame_size[0]}x{self.frame_size[1]}"
                    )
                return
            frame = cv2.resize(frame, self.frame_size, interpolation = cv2.INTER_AREA)
        self.writer.write(frame)
        self.frames_written += 1

    # Закрытие видео
    def release(self):
        if self.writer is not None:
            self.writer.release()
            if self.print_status:
                print(f"Сохранено {self.label}: {self.path} ({self.frames_written} кадров)")
            self.writer = None


# Создание кадра репроекций ключевых кадров
def build_keyframe_reprojection_video_frame(kf_canvases, grid_cols = 2, pad_color = (255, 255, 255)):
    frames = [ensure_bgr_uint8(canvas) for canvas in kf_canvases if canvas is not None]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        return None

    grid_cols = max(1, int(grid_cols))
    frames = pad_to_max_size(frames, pad_color = pad_color)
    cell_h, cell_w = frames[0].shape[:2]

    blank = np.ones((cell_h, cell_w, 3), dtype = np.uint8) * np.array(pad_color, dtype = np.uint8)
    while len(frames) % grid_cols != 0:
        frames.append(blank.copy())

    rows = []
    for start in range(0, len(frames), grid_cols):
        rows.append(np.hstack(frames[start:start + grid_cols]))
        
    return np.vstack(rows)


# Преобразование значения в JSON-совместимый формат
def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


# Преобразование в float для CSV
def _csv_float(value):
    try:
        v = float(value)
    except Exception:
        return ''
    return v if np.isfinite(v) else ''


# Преобразование в int для CSV
def _csv_int(value):
    try:
        return int(value)
    except Exception:
        return ''


# Формирование трех колонок для вектора 3D
def _vec3_columns(prefix, value):
    arr = np.asarray(value if value is not None else [np.nan, np.nan, np.nan], dtype = np.float64).reshape(-1)
    if len(arr) < 3:
        arr = np.pad(arr, (0, 3 - len(arr)), constant_values = np.nan)
    return {
        f'{prefix}_x': _csv_float(arr[0]),
        f'{prefix}_y': _csv_float(arr[1]),
        f'{prefix}_z': _csv_float(arr[2]),
    }


# Формирование двух колонок для вектора 2D
def _vec2_columns(prefix, value):
    arr = np.asarray(value if value is not None else [np.nan, np.nan], dtype = np.float64).reshape(-1)
    if len(arr) < 2:
        arr = np.pad(arr, (0, 2 - len(arr)), constant_values = np.nan)
    return {
        f'{prefix}_x': _csv_float(arr[0]),
        f'{prefix}_y': _csv_float(arr[1]),
    }


# Формирование 9 колонок для матрицы поворота
def _rotation_matrix_columns(prefix, value):
    arr = np.asarray(value if value is not None else np.full((3, 3), np.nan), dtype = np.float64).reshape(3, 3)
    return {f'{prefix}_{r}{c}': _csv_float(arr[r, c]) for r in range(3) for c in range(3)}


# Положение соседних камер
def build_pair_side_lookup(cross_pairs):
    """Tuple order in CROSS_PAIRS is treated as left->right around the camera ring."""
    side = {}
    for i1, i2, _, name in cross_pairs:
        side[(int(i1), str(name))] = 'right'
        side[(int(i2), str(name))] = 'left'
    return side


# CSV-логирование
class CsvRunLogger:
    SETTINGS_HEADERS = ['timestamp', 'run_dir', 'section', 'key', 'value_json']
    KEYFRAME_HEADERS = [
        'timestamp', 'frame_idx', 'cam_id', 'placement', 'keyframe_id', 'source', 'point_idx',
        'pixel_x', 'pixel_y',
        'unit_x', 'unit_y', 'unit_z',
        'metric_scale', 'metric_scale_applied',
        'metric_x', 'metric_y', 'metric_z',
    ]
    MOTION_POINTS_HEADERS = [
        'timestamp', 'frame_idx', 'cam_id', 'placement', 'keyframe_id', 'source', 'point_idx',
        'pixel_x', 'pixel_y',
        'input_unit_x', 'input_unit_y', 'input_unit_z',
        'refined_unit_x', 'refined_unit_y', 'refined_unit_z',
        'metric_scale', 'metric_scale_applied',
        'refined_metric_x', 'refined_metric_y', 'refined_metric_z',
    ]
    FRAME_MOTION_HEADERS = [
        'timestamp', 'frame_idx', 'cam_id', 'placement', 'state', 'is_keyframe', 'keyframe_id',
        'local_unit_x', 'local_unit_y', 'local_unit_z',
        'global_unit_x', 'global_unit_y', 'global_unit_z',
        'local_anchor_unit_x', 'local_anchor_unit_y', 'local_anchor_unit_z',
        'global_anchor_unit_x', 'global_anchor_unit_y', 'global_anchor_unit_z',
        'local_metric_x', 'local_metric_y', 'local_metric_z',
        'global_metric_x', 'global_metric_y', 'global_metric_z',
        'local_anchor_metric_x', 'local_anchor_metric_y', 'local_anchor_metric_z',
        'global_anchor_metric_x', 'global_anchor_metric_y', 'global_anchor_metric_z',
        'local_rot_x', 'local_rot_y', 'local_rot_z',
        'global_rot_x', 'global_rot_y', 'global_rot_z',
        'metric_scale', 'metric_scale_applied', 'metric_scale_source', 'metric_scale_inliers', 'metric_scale_residual',
        'unit_scale', 'init_scale_axis', 'init_scale_axis_value',
        'baseline_unit', 'baseline_rel', 'min_baseline_unit', 'min_baseline_rel',
        'motion_points_count', 'filtered_points_count', 'rotation_points_count', 'translation_points_count',
    ]
    SCALE_SUMMARY_HEADERS = [
        'timestamp', 'frame_idx', 'cam_id', 'placement',
        'left_pair_name', 'left_raw_match_count', 'left_scale_points', 'left_scale',
        'right_pair_name', 'right_raw_match_count', 'right_scale_points', 'right_scale',
        'joint_scale_estimate', 'joint_scale_points', 'joint_scale_applied',
        'metric_scale_after', 'metric_scale_ready_after', 'metric_scale_source_after',
    ]
    SCALE_POINTS_HEADERS = [
        'timestamp', 'frame_idx', 'cam_id', 'placement', 'neighbor_side', 'pair_name', 'point_idx',
        'unit_pixel_x', 'unit_pixel_y', 'stereo_pixel_x', 'stereo_pixel_y',
        'unit_x', 'unit_y', 'unit_z',
        'stereo_metric_x', 'stereo_metric_y', 'stereo_metric_z',
        'per_point_scale', 'hamming', 'reproj_error_px',
        'joint_scale_estimate', 'joint_scale_applied',
        'metric_by_point_scale_x', 'metric_by_point_scale_y', 'metric_by_point_scale_z',
        'metric_by_joint_scale_x', 'metric_by_joint_scale_y', 'metric_by_joint_scale_z',
        'metric_scale_after', 'metric_scale_ready_after',
    ]
    LOCAL_UNIT_MOTION_COUNTS_HEADERS = [
        'timestamp', 'scope', 'cam_id', 'placement', 'axis',
        'total_count', 'positive_count', 'negative_count', 'near_zero_count',
        'threshold_abs',
    ]
    LOCAL_UNIT_Z_MOTION_HEADERS = [
        'frame_idx', 'series', 'local_anchor_unit_z',
    ]
    GLOBAL_METRIC_MOTION_HEADERS = [
        'timestamp', 'frame_idx', 'scope', 'cam_id', 'placement',
        'translation_coordinate_system', 'translation_units', 'rotation_units',
        'metric_ready', 'metric_scale', 'metric_scale_source',
        'metric_sample_count_x', 'metric_sample_count_y', 'metric_sample_count_z',
        'rot_sample_count_x', 'rot_sample_count_y', 'rot_sample_count_z',
        'global_metric_x', 'global_metric_y', 'global_metric_z',
        'global_rot_x', 'global_rot_y', 'global_rot_z',
    ]

    def __init__(self, config, settings_sections = None):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get('enabled', False))
        self.csv_flags = dict(self.config.get('csv', {}))
        if 'local_unit_z_motion' not in self.csv_flags:
            self.csv_flags['local_unit_z_motion'] = bool(self.csv_flags.get('global_metric_motion', False))
        self.filenames = dict(self.config.get('filenames', {}))
        if 'local_unit_z_motion' not in self.filenames:
            self.filenames['local_unit_z_motion'] = '08_local_unit_z_motion.csv'
        try:
            self.local_unit_count_threshold = abs(float(self.config.get('local_unit_count_threshold', 0.01)))
        except Exception:
            self.local_unit_count_threshold = 0.01
        self.local_unit_motion_counts = {}
        self.run_dir = None
        self.files = {}
        self.writers = {}
        self.headers = {
            'settings': self.SETTINGS_HEADERS,
            'keyframe_points': self.KEYFRAME_HEADERS,
            'motion_points': self.MOTION_POINTS_HEADERS,
            'frame_motion': self.FRAME_MOTION_HEADERS,
            'scale_summary': self.SCALE_SUMMARY_HEADERS,
            'scale_points': self.SCALE_POINTS_HEADERS,
            'local_unit_motion_counts': self.LOCAL_UNIT_MOTION_COUNTS_HEADERS,
            'local_unit_z_motion': self.LOCAL_UNIT_Z_MOTION_HEADERS,
            'global_metric_motion': self.GLOBAL_METRIC_MOTION_HEADERS,
        }
        if self.enabled or bool(self.config.get('save_videos_in_log_dir', False)):
            root = self.config.get('root_dir', 'outputs')
            prefix = self.config.get('directory_prefix', 'log')
            fmt = self.config.get('datetime_format', '%Y%m%d_%H%M%S')
            stamp = datetime.now().strftime(fmt)
            base_run_dir = os.path.join(root, f'{prefix}_{stamp}')
            self.run_dir = base_run_dir
            suffix = 1
            while os.path.exists(self.run_dir):
                self.run_dir = f'{base_run_dir}_{suffix}'
                suffix += 1
            os.makedirs(self.run_dir, exist_ok = False)
            if self.config.get('print_status', True):
                print(f"Логи запуска: {self.run_dir}")
        if self.enabled and settings_sections is not None:
            self.log_settings(settings_sections)

    # Текущее время в ISO формате с миллисекундами
    def now(self):
        return datetime.now().isoformat(timespec = 'milliseconds')

    # Проверка, включен ли конкретный тип лога
    def is_enabled(self, name):
        return self.enabled and bool(self.csv_flags.get(name, False))

    # Определение пути для видео
    def artifact_path(self, configured_path, default_name):
        if self.run_dir and bool(self.config.get('save_videos_in_log_dir', False)):
            name = os.path.basename(configured_path or default_name) or default_name
            return os.path.join(self.run_dir, name)
        
        return configured_path or default_name

    # Открытие CSV-файла для записи
    def _open_writer(self, name):
        if name in self.writers:
            return self.writers[name]
        if not self.is_enabled(name):
            return None
        if self.run_dir is None:
            root = self.config.get('root_dir', 'outputs')
            prefix = self.config.get('directory_prefix', 'log')
            fmt = self.config.get('datetime_format', '%Y%m%d_%H%M%S')
            base_run_dir = os.path.join(root, f"{prefix}_{datetime.now().strftime(fmt)}")
            self.run_dir = base_run_dir
            suffix = 1
            while os.path.exists(self.run_dir):
                self.run_dir = f'{base_run_dir}_{suffix}'
                suffix += 1
            os.makedirs(self.run_dir, exist_ok = False)
        filename = self.filenames.get(name, f'{name}.csv')
        path = os.path.join(self.run_dir, filename)
        f = open(path, 'w', newline = '', encoding = 'utf-8')
        writer = csv.DictWriter(f, fieldnames = self.headers[name], extrasaction = 'ignore')
        writer.writeheader()
        self.files[name] = f
        self.writers[name] = writer
        if self.config.get('print_status', True):
            print(f"CSV лог создан: {path}")
            
        return writer

    # Запись одной строки в CSV-файл
    def write_row(self, name, row):
        writer = self._open_writer(name)
        if writer is None:
            return
        writer.writerow(row)
        self.files[name].flush()

    # Логирование настроек в CSV
    def log_settings(self, sections):
        if not self.is_enabled('settings'):
            return
        ts = self.now()

        def walk(section, value, prefix = ''):
            if isinstance(value, dict):
                for key, child in value.items():
                    child_key = f'{prefix}.{key}' if prefix else str(key)
                    yield from walk(section, child, child_key)
            elif isinstance(value, (list, tuple)) and value and all(isinstance(v, dict) for v in value):
                for idx, child in enumerate(value):
                    child_key = f'{prefix}[{idx}]' if prefix else f'[{idx}]'
                    yield from walk(section, child, child_key)
            else:
                yield section, prefix, json.dumps(_json_safe(value), ensure_ascii = False)

        for section, value in sections.items():
            for sec, key, value_json in walk(section, value):
                self.write_row('settings', {
                    'timestamp': ts,
                    'run_dir': self.run_dir or '',
                    'section': sec,
                    'key': key,
                    'value_json': value_json,
                })

    # Логирование точек ключевых кадров
    def log_keyframe_events(self, trackers):
        if not self.is_enabled('keyframe_points'):
            return
        for tr in trackers:
            event = getattr(tr, 'last_keyframe_event', None) if tr is not None else None
            if not event:
                continue
            ts = self.now()
            pts3d = np.asarray(event.get('pts3d_unit', np.empty((0, 3))), dtype = np.float64).reshape(-1, 3)
            pts2d = np.asarray(event.get('pts2d', np.empty((0, 2))), dtype = np.float64).reshape(-1, 2)
            n = min(len(pts3d), len(pts2d)) if len(pts2d) else len(pts3d)
            metric_ready = bool(getattr(tr, 'has_metric_scale', lambda: False)())
            metric_scale = float(getattr(tr, 'metric_scale', np.nan)) if metric_ready else np.nan
            for idx in range(n):
                p_unit = pts3d[idx]
                p_metric = p_unit * metric_scale if metric_ready else np.full(3, np.nan, dtype = np.float64)
                row = {
                    'timestamp': ts,
                    'frame_idx': _csv_int(event.get('frame_idx')),
                    'cam_id': _csv_int(event.get('cam_id')),
                    'placement': event.get('placement', ''),
                    'keyframe_id': _csv_int(event.get('keyframe_id')),
                    'source': event.get('source', ''),
                    'point_idx': idx,
                    'metric_scale': _csv_float(metric_scale),
                    'metric_scale_applied': metric_ready,
                }
                row.update(_vec2_columns('pixel', pts2d[idx] if idx < len(pts2d) else None))
                row.update(_vec3_columns('unit', p_unit))
                row.update(_vec3_columns('metric', p_metric))
                self.write_row('keyframe_points', row)
            tr.last_keyframe_event = None

    # Логирование точек, участвующих в оценке движения
    def log_motion_points(self, trackers):
        if not self.is_enabled('motion_points'):
            return
        for tr in trackers:
            event = getattr(tr, 'last_motion_points_for_logging', None) if tr is not None else None
            if not event:
                continue
            if getattr(tr, 'state', None) != 'TRACKING' or int(event.get('frame_idx', -1)) != int(getattr(tr, 'frame_idx', -2)):
                continue
            ts = self.now()
            pixels = np.asarray(event.get('pixels', np.empty((0, 2))), dtype = np.float64).reshape(-1, 2)
            p_input = np.asarray(event.get('pts3d_unit_input', np.empty((0, 3))), dtype = np.float64).reshape(-1, 3)
            p_refined = np.asarray(event.get('pts3d_unit_refined', np.empty((0, 3))), dtype = np.float64).reshape(-1, 3)
            n = min(len(pixels), len(p_input), len(p_refined))
            metric_ready = bool(getattr(tr, 'has_metric_scale', lambda: False)())
            metric_scale = float(getattr(tr, 'metric_scale', np.nan)) if metric_ready else np.nan
            for idx in range(n):
                refined_metric = p_refined[idx] * metric_scale if metric_ready else np.full(3, np.nan, dtype = np.float64)
                row = {
                    'timestamp': ts,
                    'frame_idx': _csv_int(event.get('frame_idx')),
                    'cam_id': _csv_int(event.get('cam_id')),
                    'placement': event.get('placement', ''),
                    'keyframe_id': _csv_int(event.get('keyframe_id')),
                    'source': event.get('source', ''),
                    'point_idx': idx,
                    'metric_scale': _csv_float(metric_scale),
                    'metric_scale_applied': metric_ready,
                }
                row.update(_vec2_columns('pixel', pixels[idx]))
                row.update(_vec3_columns('input_unit', p_input[idx]))
                row.update(_vec3_columns('refined_unit', p_refined[idx]))
                row.update(_vec3_columns('refined_metric', refined_metric))
                self.write_row('motion_points', row)

    # Логирование движения по каждому кадру
    def log_frame_motion(self, trackers):
        if not self.is_enabled('frame_motion'):
            return
        ts = self.now()
        for tr in trackers:
            if tr is None or getattr(tr, 'state', None) != 'TRACKING':
                continue
            motion = getattr(tr, 'last_motion', None)
            if not motion or int(motion.get('frame_idx', -1)) != int(getattr(tr, 'frame_idx', -2)):
                continue
            row = {
                'timestamp': ts,
                'frame_idx': _csv_int(motion.get('frame_idx')),
                'cam_id': _csv_int(motion.get('cam_id')),
                'placement': motion.get('placement', ''),
                'state': motion.get('state', ''),
                'is_keyframe': bool(motion.get('is_keyframe', False)),
                'keyframe_id': _csv_int(motion.get('keyframe_id')),
                'metric_scale': _csv_float(motion.get('metric_scale')),
                'metric_scale_applied': bool(motion.get('metric_ready', False)),
                'metric_scale_source': motion.get('metric_scale_source', ''),
                'metric_scale_inliers': _csv_int(motion.get('metric_scale_inliers')),
                'metric_scale_residual': _csv_float(motion.get('metric_scale_residual')),
                'unit_scale': _csv_float(motion.get('unit_scale')),
                'init_scale_axis': _csv_int(motion.get('init_scale_axis')),
                'init_scale_axis_value': _csv_float(motion.get('init_scale_axis_value')),
                'baseline_unit': _csv_float(motion.get('baseline_unit')),
                'baseline_rel': _csv_float(motion.get('baseline_rel')),
                'min_baseline_unit': _csv_float(motion.get('min_baseline_unit')),
                'min_baseline_rel': _csv_float(motion.get('min_baseline_rel')),
                'motion_points_count': _csv_int(motion.get('motion_points_count', motion.get('inliers'))),
                'filtered_points_count': _csv_int(motion.get('filtered_points_count', motion.get('inliers'))),
                'rotation_points_count': _csv_int(motion.get('rotation_points_count', motion.get('inliers'))),
                'translation_points_count': _csv_int(motion.get('translation_points_count', motion.get('inliers'))),
            }
            for key, prefix in [
                ('local_t_unit', 'local_unit'), ('global_t_unit', 'global_unit'),
                ('local_t_anchor_unit', 'local_anchor_unit'), ('global_t_anchor_unit', 'global_anchor_unit'),
                ('local_t_metric', 'local_metric'), ('global_t_metric', 'global_metric'),
                ('local_t_anchor_metric', 'local_anchor_metric'), ('global_t_anchor_metric', 'global_anchor_metric'),
                ('local_rot', 'local_rot'), ('global_rot', 'global_rot'),
            ]:
                row.update(_vec3_columns(prefix, motion.get(key)))
            self.write_row('frame_motion', row)

    # Логирование данных о метрическом масштабе
    def log_scale_summary(self, trackers, summary_by_cam):
        if not self.is_enabled('scale_summary'):
            return
        ts = self.now()
        for tr in trackers:
            if tr is None or getattr(tr, 'state', None) != 'TRACKING':
                continue
            cam_id = int(getattr(tr, 'cam_id', -1))
            frame_idx = int(getattr(tr, 'frame_idx', -1))
            data = summary_by_cam.get(cam_id, {})
            left = data.get('left', {})
            right = data.get('right', {})
            metric_ready = bool(getattr(tr, 'has_metric_scale', lambda: False)())
            self.write_row('scale_summary', {
                'timestamp': ts,
                'frame_idx': frame_idx,
                'cam_id': cam_id,
                'placement': getattr(tr, 'placement', ''),
                'left_pair_name': left.get('pair_name', ''),
                'left_raw_match_count': _csv_int(left.get('raw_match_count', 0)),
                'left_scale_points': _csv_int(left.get('scale_points', 0)),
                'left_scale': _csv_float(left.get('scale')),
                'right_pair_name': right.get('pair_name', ''),
                'right_raw_match_count': _csv_int(right.get('raw_match_count', 0)),
                'right_scale_points': _csv_int(right.get('scale_points', 0)),
                'right_scale': _csv_float(right.get('scale')),
                'joint_scale_estimate': _csv_float(data.get('joint_scale')),
                'joint_scale_points': _csv_int(data.get('joint_scale_points', 0)),
                'joint_scale_applied': bool(data.get('joint_scale_applied', False)),
                'metric_scale_after': _csv_float(getattr(tr, 'metric_scale', np.nan) if metric_ready else np.nan),
                'metric_scale_ready_after': metric_ready,
                'metric_scale_source_after': getattr(tr, 'metric_scale_source', '') or '',
            })

    # Логирование масштабов каждой точки
    def log_scale_points(self, tracker, estimate, joint_scale_applied, side_lookup):
        if not self.is_enabled('scale_points') or tracker is None or estimate is None:
            return
        details = estimate.get('point_details') or {}
        unit_points = np.asarray(details.get('unit_points', np.empty((0, 3))), dtype = np.float64).reshape(-1, 3)
        metric_points = np.asarray(details.get('metric_points', np.empty((0, 3))), dtype = np.float64).reshape(-1, 3)
        unit_pixels = np.asarray(details.get('unit_pixels', np.empty((0, 2))), dtype = np.float64).reshape(-1, 2)
        stereo_pixels = np.asarray(details.get('stereo_pixels', np.empty((0, 2))), dtype = np.float64).reshape(-1, 2)
        scales = np.asarray(details.get('scales', np.empty(0)), dtype = np.float64).reshape(-1)
        pair_names = np.asarray(details.get('pair_names', np.empty(0, dtype = object)), dtype = object).reshape(-1)
        hamming = np.asarray(details.get('hamming', np.empty(0)), dtype = np.float64).reshape(-1)
        reproj = np.asarray(details.get('reproj_error', np.empty(0)), dtype = np.float64).reshape(-1)
        n = min(len(unit_points), len(metric_points), len(unit_pixels), len(stereo_pixels), len(scales), len(pair_names))
        if n <= 0:
            return
        ts = self.now()
        cam_id = int(getattr(tracker, 'cam_id', -1))
        metric_ready_after = bool(getattr(tracker, 'has_metric_scale', lambda: False)())
        metric_scale_after = float(getattr(tracker, 'metric_scale', np.nan)) if metric_ready_after else np.nan
        joint_scale = float(estimate.get('scale', np.nan))
        for idx in range(n):
            pair_name = str(pair_names[idx])
            per_point_scale = float(scales[idx])
            metric_by_point = unit_points[idx] * per_point_scale
            metric_by_joint = unit_points[idx] * joint_scale if np.isfinite(joint_scale) else np.full(3, np.nan)
            row = {
                'timestamp': ts,
                'frame_idx': _csv_int(getattr(tracker, 'frame_idx', None)),
                'cam_id': cam_id,
                'placement': getattr(tracker, 'placement', ''),
                'neighbor_side': side_lookup.get((cam_id, pair_name), ''),
                'pair_name': pair_name,
                'point_idx': idx,
                'per_point_scale': _csv_float(per_point_scale),
                'hamming': _csv_float(hamming[idx] if idx < len(hamming) else np.nan),
                'reproj_error_px': _csv_float(reproj[idx] if idx < len(reproj) else np.nan),
                'joint_scale_estimate': _csv_float(joint_scale),
                'joint_scale_applied': bool(joint_scale_applied),
                'metric_scale_after': _csv_float(metric_scale_after),
                'metric_scale_ready_after': metric_ready_after,
            }
            row.update(_vec2_columns('unit_pixel', unit_pixels[idx]))
            row.update(_vec2_columns('stereo_pixel', stereo_pixels[idx]))
            row.update(_vec3_columns('unit', unit_points[idx]))
            row.update(_vec3_columns('stereo_metric', metric_points[idx]))
            row.update(_vec3_columns('metric_by_point_scale', metric_by_point))
            row.update(_vec3_columns('metric_by_joint_scale', metric_by_joint))
            self.write_row('scale_points', row)

    # Создание переменных для подсчёта наличия и направления движения
    def _local_count_bucket(self, scope, cam_id, placement):
        key = (str(scope), '' if cam_id is None else int(cam_id), str(placement or ''))
        if key not in self.local_unit_motion_counts:
            self.local_unit_motion_counts[key] = {
                'scope': key[0],
                'cam_id': key[1],
                'placement': key[2],
                'counts': [
                    {'total_count': 0, 'positive_count': 0, 'negative_count': 0, 'near_zero_count': 0},
                    {'total_count': 0, 'positive_count': 0, 'negative_count': 0, 'near_zero_count': 0},
                    {'total_count': 0, 'positive_count': 0, 'negative_count': 0, 'near_zero_count': 0},
                ],
                'seen_samples': set(),
            }
        return self.local_unit_motion_counts[key]

    # Подсчёта наличия и направлеия движения
    def _count_local_unit_vector(self, scope, cam_id, placement, values, sample_id):
        arr = np.asarray(values if values is not None else [np.nan, np.nan, np.nan], dtype = np.float64).reshape(-1)
        if len(arr) < 3:
            arr = np.pad(arr, (0, 3 - len(arr)), constant_values = np.nan)
        bucket = self._local_count_bucket(scope, cam_id, placement)
        if sample_id in bucket['seen_samples']:
            return
        bucket['seen_samples'].add(sample_id)
        threshold = float(self.local_unit_count_threshold)
        for axis_idx in range(3):
            v = float(arr[axis_idx])
            if not np.isfinite(v):
                continue
            counts = bucket['counts'][axis_idx]
            counts['total_count'] += 1
            if v >= threshold:
                counts['positive_count'] += 1
            elif v <= -threshold:
                counts['negative_count'] += 1
            else:
                counts['near_zero_count'] += 1

    # Обновление счетчиков движения в локальной системе координат
    def update_local_unit_motion_counts(self, motion_states, avg_data = None):

        if not self.is_enabled('local_unit_motion_counts'):
            return
        motion_states = list(motion_states or [])
        for motion in motion_states:
            if not motion:
                continue
            values = motion.get('local_t_anchor_unit')
            if values is None:
                continue
            cam_id = motion.get('cam_id')
            frame_idx = motion.get('frame_idx')
            sample_id = ('camera', cam_id, frame_idx)
            self._count_local_unit_vector(
                'camera', cam_id, motion.get('placement', ''), values, sample_id
            )

        if avg_data is None:
            return
        avg_local = (avg_data.get('local_t_unit') or {}).get('avg')
        if avg_local is None:
            return
        avg_frame_key = tuple(
            sorted((int(m.get('cam_id', -1)), int(m.get('frame_idx', -1))) for m in motion_states if m)
        )
        if not avg_frame_key:
            return
        self._count_local_unit_vector(
            'average', None, 'average', avg_local, ('average', avg_frame_key)
        )

    # Логирование накопленных счетчиков движения в CSV
    def flush_local_unit_motion_counts(self):
        if not self.is_enabled('local_unit_motion_counts') or not self.local_unit_motion_counts:
            return
        ts = self.now()
        axis_names = ('x', 'y', 'z')

        def sort_key(item):
            (scope, cam_id, placement), _ = item
            cam_sort = 10**9 if cam_id  == '' else int(cam_id)
            return (0 if scope  == 'camera' else 1, cam_sort, placement)

        for _, bucket in sorted(self.local_unit_motion_counts.items(), key = sort_key):
            for axis_idx, axis_name in enumerate(axis_names):
                counts = bucket['counts'][axis_idx]
                self.write_row('local_unit_motion_counts', {
                    'timestamp': ts,
                    'scope': bucket['scope'],
                    'cam_id': bucket['cam_id'],
                    'placement': bucket['placement'],
                    'axis': axis_name,
                    'total_count': int(counts['total_count']),
                    'positive_count': int(counts['positive_count']),
                    'negative_count': int(counts['negative_count']),
                    'near_zero_count': int(counts['near_zero_count']),
                    'threshold_abs': _csv_float(self.local_unit_count_threshold),
                })

    # Логирование движения по оси Z в локальной системе координат
    def log_local_unit_z_motion(self, motion_states, avg_data = None):
        if not self.is_enabled('local_unit_z_motion'):
            return
        motion_states = list(motion_states or [])
        if not motion_states and avg_data is None:
            return

        frame_ids = []

        def z_value(values):
            arr = np.asarray(values if values is not None else [np.nan, np.nan, np.nan], dtype = np.float64).reshape(-1)
            if len(arr) < 3:
                arr = np.pad(arr, (0, 3 - len(arr)), constant_values = np.nan)
            return arr[2]

        for motion in motion_states:
            if not motion:
                continue
            frame_idx = _csv_int(motion.get('frame_idx'))
            if frame_idx != '':
                frame_ids.append(int(frame_idx))
            cam_id = _csv_int(motion.get('cam_id'))
            values = motion.get('local_t_anchor_unit')
            if values is None:
                continue
            self.write_row('local_unit_z_motion', {
                'frame_idx': frame_idx,
                'series': f'cam_{cam_id}',
                'local_anchor_unit_z': _csv_float(z_value(values)),
            })

        if avg_data is None:
            return
        avg_local = (avg_data.get('local_t_unit') or {}).get('avg')
        if avg_local is None:
            return
        self.write_row('local_unit_z_motion', {
            'frame_idx': max(frame_ids) if frame_ids else '',
            'series': 'average',
            'local_anchor_unit_z': _csv_float(z_value(avg_local)),
        })

    # Логирование глобального метрического движения
    def log_global_metric_motion(self, motion_states, avg_data = None):
        if not self.is_enabled('global_metric_motion'):
            return
        motion_states = list(motion_states or [])
        if not motion_states and avg_data is None:
            return
        ts = self.now()

        def finite_count_vec(values):
            arr = np.asarray(values if values is not None else [np.nan, np.nan, np.nan], dtype = np.float64).reshape(-1)
            if len(arr) < 3:
                arr = np.pad(arr, (0, 3 - len(arr)), constant_values = np.nan)
            return [int(np.isfinite(arr[i])) for i in range(3)]

        def count_columns(prefix, counts):
            counts = list(counts) if counts is not None else [0, 0, 0]
            if len(counts) < 3:
                counts = counts + [0] * (3 - len(counts))
            return {
                f'{prefix}_sample_count_x': _csv_int(counts[0]),
                f'{prefix}_sample_count_y': _csv_int(counts[1]),
                f'{prefix}_sample_count_z': _csv_int(counts[2]),
            }

        frame_ids = []
        for motion in motion_states:
            if not motion:
                continue
            frame_idx = _csv_int(motion.get('frame_idx'))
            if frame_idx != '':
                frame_ids.append(int(frame_idx))
            global_metric = motion.get('global_t_anchor_metric')
            if global_metric is None:
                global_metric = motion.get('global_t_metric')
            global_rot = motion.get('global_rot')
            row = {
                'timestamp': ts,
                'frame_idx': frame_idx,
                'scope': 'camera',
                'cam_id': _csv_int(motion.get('cam_id')),
                'placement': motion.get('placement', ''),
                'translation_coordinate_system': 'anchor',
                'translation_units': 'm',
                'rotation_units': 'deg',
                'metric_ready': bool(motion.get('metric_ready', False)),
                'metric_scale': _csv_float(motion.get('metric_scale')),
                'metric_scale_source': motion.get('metric_scale_source', ''),
            }
            row.update(count_columns('metric', finite_count_vec(global_metric)))
            row.update(count_columns('rot', finite_count_vec(global_rot)))
            row.update(_vec3_columns('global_metric', global_metric))
            row.update(_vec3_columns('global_rot', global_rot))
            self.write_row('global_metric_motion', row)

        if avg_data is None:
            return
        avg_metric_data = avg_data.get('global_t_metric') or {}
        avg_rot_data = avg_data.get('global_rot') or {}
        avg_metric = avg_metric_data.get('avg')
        avg_rot = avg_rot_data.get('avg')
        if avg_metric is None and avg_rot is None:
            return
        metric_counts = avg_metric_data.get('counts', finite_count_vec(avg_metric))
        rot_counts = avg_rot_data.get('counts', finite_count_vec(avg_rot))
        row = {
            'timestamp': ts,
            'frame_idx': max(frame_ids) if frame_ids else '',
            'scope': 'average',
            'cam_id': '',
            'placement': 'average',
            'translation_coordinate_system': 'anchor',
            'translation_units': 'm',
            'rotation_units': 'deg',
            'metric_ready': '',
            'metric_scale': '',
            'metric_scale_source': '',
        }
        row.update(count_columns('metric', metric_counts))
        row.update(count_columns('rot', rot_counts))
        row.update(_vec3_columns('global_metric', avg_metric))
        row.update(_vec3_columns('global_rot', avg_rot))
        self.write_row('global_metric_motion', row)

    # Закрытие всех CSV-файлов
    def close(self):
        self.flush_local_unit_motion_counts()
        for f in list(self.files.values()):
            try:
                f.close()
            except Exception:
                pass
        self.files.clear()
        self.writers.clear()


# Обновление количества совпадений для пары камер
def update_scale_pair_count(summary_by_cam, cam_a, cam_b, pair_name, count):
    summary_by_cam.setdefault(int(cam_a), {}).setdefault('right', {'pair_name': pair_name})
    summary_by_cam.setdefault(int(cam_b), {}).setdefault('left', {'pair_name': pair_name})
    summary_by_cam[int(cam_a)]['right'].update({'pair_name': pair_name, 'raw_match_count': int(count)})
    summary_by_cam[int(cam_b)]['left'].update({'pair_name': pair_name, 'raw_match_count': int(count)})


# Обновление оценки масштаба для пары камер
def update_scale_pair_estimate(summary_by_cam, cam_id, pair_name, side_lookup, scale, points):
    side = side_lookup.get((int(cam_id), str(pair_name)), '')
    if side not in ('left', 'right'):
        return
    summary_by_cam.setdefault(int(cam_id), {}).setdefault(side, {'pair_name': pair_name})
    summary_by_cam[int(cam_id)][side].update({
        'pair_name': str(pair_name),
        'scale': float(scale) if np.isfinite(scale) else np.nan,
        'scale_points': int(points),
    })


def main():
    # Настройка максимального количества камер    
    MAX_CAMS = int(APP_CONFIG.get('max_cams', len(CONFIGS)))
    DEFAULT_FRAME_SIZE = tuple(APP_CONFIG.get('default_frame_size', (480, 640)))

    # Построение поворотов относительно якорной камеры
    anchor_rotations = build_camera_to_anchor_rotations(CROSS_CALIB, anchor = APP_CONFIG.get('anchor_camera_id', 1))

    # Построение словаря сторон для стереопар
    pair_side_lookup = build_pair_side_lookup(CROSS_PAIRS)

    # Создание логгера
    run_logger = CsvRunLogger(LOGGING_CONFIG, settings_sections = None)

    # Создание трекеров для каждой камеры
    trackers = [None] * MAX_CAMS
    for i in range(min(MAX_CAMS, len(CONFIGS))):
        try:
            tracker_config = dict(CONFIGS[i])
            tracker_config.update({
                'video_panel_width': DISPLAY_CONFIG['video_panel_width'],
                'kf_panel_width': DISPLAY_CONFIG['kf_panel_width'],
                'display_font_scale': DISPLAY_CONFIG['font_scale'],
                'display_font_thickness': DISPLAY_CONFIG['font_thickness'],
                'display_line_height': DISPLAY_CONFIG['line_height'],
                'display_section_gap': DISPLAY_CONFIG['section_gap'],
                'display_margin_x': DISPLAY_CONFIG['text_margin_x'],
                'display_y_start': DISPLAY_CONFIG['text_y_start'],
                'R_cam_to_anchor': anchor_rotations.get(i, np.eye(3, dtype = np.float64)),
            })
            tr = CameraTracker(tracker_config, i, show_tracking_points = DISPLAY_CONFIG['show_tracking_points'])
            trackers[i] = tr
        except Exception as e:
            print(f"Ошибка при создании трекера для камеры {i}: {e}")

    active = [tr is not None for tr in trackers]

    # Создание окна для отображения
    cv2.namedWindow(DISPLAY_CONFIG['window_video'])

    # Настройка FPS для записи
    recording_fps = resolve_recording_fps(trackers, RECORDING_CONFIG)

    # Определение путей для видеофайлов
    final_window_path = run_logger.artifact_path(
        RECORDING_CONFIG.get('final_window_path', 'outputs/final_window.avi'),
        'final_window.avi'
    )
    keyframe_reprojection_path = run_logger.artifact_path(
        RECORDING_CONFIG.get('keyframe_reprojection_path', 'outputs/keyframe_reprojection.avi'),
        'keyframe_reprojection.avi'
    )

    # Логирование настроек
    run_logger.log_settings({
        'APP_CONFIG': APP_CONFIG,
        'ORB_DEFAULTS': ORB_DEFAULTS,
        'TRACKER_DEFAULTS': TRACKER_DEFAULTS,
        'MODEL_ESTIMATION_DEFAULTS': MODEL_ESTIMATION_DEFAULTS,
        'EVALUATE_PAIR_CONFIG': EVALUATE_PAIR_CONFIG,
        'POSE_MATCH_CONFIG': POSE_MATCH_CONFIG,
        'PNP_CONFIG': PNP_CONFIG,
        'KEYFRAME_CONFIG': KEYFRAME_CONFIG,
        'DISPLAY_CONFIG': DISPLAY_CONFIG,
        'METRIC_SCALE_CONFIG': METRIC_SCALE_CONFIG,
        'CROSS_MATCH_CONFIG': CROSS_MATCH_CONFIG,
        'GEOMETRY_CONFIG': GEOMETRY_CONFIG,
        'RECORDING_CONFIG': RECORDING_CONFIG,
        'LOGGING_CONFIG': LOGGING_CONFIG,
        'CAMERA_CONFIGS': CONFIGS,
        'CROSS_PAIRS': CROSS_PAIRS,
        'CROSS_CALIB': CROSS_CALIB,
        'ANCHOR_ROTATIONS': anchor_rotations,
        'EFFECTIVE_OUTPUTS': {
            'run_dir': run_logger.run_dir,
            'final_window_path': final_window_path,
            'keyframe_reprojection_path': keyframe_reprojection_path,
            'recording_fps': recording_fps,
        }
    })

    # Создание объектов для записи видео
    final_window_recorder = LazyVideoRecorder(
        RECORDING_CONFIG.get('save_final_window', False),
        final_window_path,
        recording_fps,
        fourcc = RECORDING_CONFIG.get('fourcc', 'XVID'),
        resize_on_size_change = RECORDING_CONFIG.get('resize_on_size_change', True),
        print_status = RECORDING_CONFIG.get('print_status', True),
        label = 'финального окна'
    )
    kf_reprojection_recorder = LazyVideoRecorder(
        RECORDING_CONFIG.get('save_keyframe_reprojection', False),
        keyframe_reprojection_path,
        recording_fps,
        fourcc = RECORDING_CONFIG.get('fourcc', 'XVID'),
        resize_on_size_change = RECORDING_CONFIG.get('resize_on_size_change', True),
        print_status = RECORDING_CONFIG.get('print_status', True),
        label = 'репроекций ключевых кадров'
    )

    # Основной цикл обработки кадров
    while any(active):
        # Буферы для холстов
        video_canvases = [None] * MAX_CAMS
        kf_canvases = [None] * MAX_CAMS

        # Чтение и обработка кадров со всех камер
        for i in range(MAX_CAMS):
            if active[i] and trackers[i] is not None:
                ret, frame = trackers[i].read_frame()
                if not ret:
                    active[i] = False
                    video_canvases[i] = create_dummy_canvas(
                        DEFAULT_FRAME_SIZE, i, DISPLAY_CONFIG['video_panel_width'],
                        "video ended", DISPLAY_CONFIG['camera_label_font_scale'],
                        DISPLAY_CONFIG['font_thickness'])
                    kf_canvases[i] = create_dummy_canvas(
                        DEFAULT_FRAME_SIZE, i, DISPLAY_CONFIG['kf_panel_width'],
                        "no KF", DISPLAY_CONFIG['camera_label_font_scale'],
                        DISPLAY_CONFIG['font_thickness'])
                    continue
                video_canvas, kf_canvas, status = trackers[i].process_frame(frame)
                if video_canvas is None:
                    video_canvas = create_dummy_canvas(
                        frame.shape, i, DISPLAY_CONFIG['video_panel_width'], status,
                        DISPLAY_CONFIG['camera_label_font_scale'],
                        DISPLAY_CONFIG['font_thickness'])
                if kf_canvas is None:
                    kf_canvas = create_dummy_canvas(
                        frame.shape, i, DISPLAY_CONFIG['kf_panel_width'], "no KF",
                        DISPLAY_CONFIG['camera_label_font_scale'],
                        DISPLAY_CONFIG['font_thickness'])
                video_canvases[i] = video_canvas.astype(np.uint8)
                kf_canvases[i] = kf_canvas.astype(np.uint8)
            else:
                video_canvases[i] = create_dummy_canvas(
                    DEFAULT_FRAME_SIZE, i, DISPLAY_CONFIG['video_panel_width'],
                    "no camera", DISPLAY_CONFIG['camera_label_font_scale'],
                    DISPLAY_CONFIG['font_thickness'])
                kf_canvases[i] = create_dummy_canvas(
                    DEFAULT_FRAME_SIZE, i, DISPLAY_CONFIG['kf_panel_width'],
                    "no camera", DISPLAY_CONFIG['camera_label_font_scale'],
                    DISPLAY_CONFIG['font_thickness'])

        # Запуск синхронной инициализации, если все камеры готовы
        maybe_start_synchronous_initialization(
            trackers, active, video_canvases, kf_canvases, DEFAULT_FRAME_SIZE
        )

        # Сбор видео-слоев для наложения поверх панелей
        frame_layers = []
        for i, tr in enumerate(trackers):
            if active[i] and tr is not None:
                frame_layers.append(tr.get_current_video_frame(copy = True))
            else:
                frame_layers.append(None)

        # Обработка стереопар
        match_counts = {}
        scale_summary_by_cam = {}
        stereo_observations = {}
        cross_visual_observations = {}
        for i1, i2, color, name in CROSS_PAIRS:
            if not active[i1] or not active[i2]:
                continue
            if trackers[i1].state != 'TRACKING' or trackers[i2].state != 'TRACKING':
                match_counts[name] = 0
                continue
            if frame_layers[i1] is None or frame_layers[i2] is None:
                continue
            
            # Получение точек с обеих камер
            features1 = trackers[i1].get_current_features(copy = False)
            features2 = trackers[i2].get_current_features(copy = False)
            pts1_pix, pts2_pix, matches = trackers[i1].orb.match_feature_sets(features1, features2)

            if len(matches) < 4:
                match_counts[name] = 0
                continue

            # Фильтрация по расстоянию Хэмминга
            hamming_cross_thresh = CROSS_MATCH_CONFIG.get('hamming_threshold', 200)
            keep = np.fromiter(
                (m.distance <= hamming_cross_thresh for m in matches),
                dtype = bool,
                count = len(matches)
            )
            pts1_pix = pts1_pix[keep]
            pts2_pix = pts2_pix[keep]
            matches = [m for m, ok in zip(matches, keep) if ok]
            if len(matches) < 4:
                match_counts[name] = 0
                continue

            pts1_arr = np.asarray(pts1_pix, dtype = np.float64)
            pts2_arr = np.asarray(pts2_pix, dtype = np.float64)

            if not GEOMETRY_CONFIG.get('use_geometric_filter', True):
                match_counts[name] = len(pts1_arr)
                if DISPLAY_CONFIG.get('show_cross_camera_points', True):
                    for p1, p2 in zip(pts1_arr, pts2_arr):
                        append_cross_camera_visual_observation(cross_visual_observations, i1, p1, color)
                        append_cross_camera_visual_observation(cross_visual_observations, i2, p2, color)
                continue

            # Геометрическая фильтрация
            cam1 = trackers[i1].camera
            cam2 = trackers[i2].camera

            rays1 = cam1.pixel_to_ray_vectorized(pts1_arr)
            rays2 = cam2.pixel_to_ray_vectorized(pts2_arr)
            norm1 = cam1.ray_to_plane_vectorized(rays1)
            norm2 = cam2.ray_to_plane_vectorized(rays2)

            key = (i1, i2)
            calib = CROSS_CALIB.get(key)
            if calib is None:
                print(f"Нет калибровки для пары {key}")
                match_counts[name] = 0
                continue
            R_rel = calib['R']
            t_rel = calib['t']

            epi_mask = essential_filter(
                norm1, norm2, R_rel, t_rel,
                threshold = GEOMETRY_CONFIG.get('epipolar_threshold', 7.0)
            )
            if np.sum(epi_mask) < 2:
                match_counts[name] = 0
                continue

            pts1_arr_good = pts1_arr[epi_mask]
            pts2_arr_good = pts2_arr[epi_mask]
            rays1_good = rays1[epi_mask]
            rays2_good = rays2[epi_mask]
            matches_epi = [m for m, ok in zip(matches, epi_mask) if ok]

            # Триангуляция и проверка глубины
            depth_mask, points3d_cam1 = triangulate_and_check_depth(rays1_good, rays2_good, R_rel, t_rel)
            final_mask = depth_mask
            final_pts1 = pts1_arr_good[final_mask]
            final_pts2 = pts2_arr_good[final_mask]
            points3d_final = points3d_cam1[final_mask]
            matches_final = [m for m, ok in zip(matches_epi, final_mask) if ok]

            match_counts[name] = len(final_pts1)

            metric_factor = float(METRIC_SCALE_CONFIG.get('calib_unit_to_meter', 1.0))
            if len(points3d_final) and len(matches_final):
                idx1_final = np.fromiter((m.queryIdx for m in matches_final), dtype = np.int32, count = len(matches_final))
                idx2_final = np.fromiter((m.trainIdx for m in matches_final), dtype = np.int32, count = len(matches_final))
                desc1_final = features1.descriptors[idx1_final]
                desc2_final = features2.descriptors[idx2_final]

                points3d_cam2_final = (R_rel @ points3d_final.T).T + t_rel.reshape(1, 3)
                append_stereo_observation(stereo_observations, i1, final_pts1, points3d_final * metric_factor, desc1_final, name)
                append_stereo_observation(stereo_observations, i2, final_pts2, points3d_cam2_final * metric_factor, desc2_final, name)

            # Визуализация точек с соседних камер
            if DISPLAY_CONFIG.get('show_cross_camera_points', True):
                draw_coords = DISPLAY_CONFIG.get('show_cross_camera_coords', True)
                for p1, p2, pt3d in zip(final_pts1, final_pts2, points3d_final):
                    coord_text = None
                    if draw_coords:
                        pt3d_m = pt3d * float(METRIC_SCALE_CONFIG.get('calib_unit_to_meter', 1.0))
                        coord_text = f"({pt3d_m[0]:.2f},{pt3d_m[1]:.2f},{pt3d_m[2]:.2f})m"
                    append_cross_camera_visual_observation(cross_visual_observations, i1, p1, color, coord_text)
                    append_cross_camera_visual_observation(cross_visual_observations, i2, p2, color, coord_text)

        # Обновление счетчиков совпадений
        for i1, i2, _, name in CROSS_PAIRS:
            update_scale_pair_count(scale_summary_by_cam, i1, i2, name, match_counts.get(name, 0))

        # Оценка метрического масштаба по точкам с соседних камер
        for cam_id, obs_raw in stereo_observations.items():
            if cam_id >= len(trackers) or trackers[cam_id] is None or trackers[cam_id].state != 'TRACKING':
                continue
            obs = merge_stereo_observation(obs_raw)
            if obs is None:
                continue
            obs_pixels = obs['pixels']
            obs_points3d = obs['points3d']
            obs_descriptors = obs['descriptors']
            obs_pair_names = obs['pair_names']
            if len(obs_pixels) < METRIC_SCALE_CONFIG.get('min_scale_points', 6):
                continue

            estimate = estimate_metric_scale_simple(
                trackers[cam_id],
                obs_pixels,
                obs_points3d,
                obs_descriptors,
                obs_pair_names,
                METRIC_SCALE_CONFIG
            )
            if estimate is None:
                continue

            # Обновление метрического масштаба в трекере
            updated = trackers[cam_id].update_metric_scale(
                estimate['scale'],
                source = obs['source'],
                inliers = estimate['points'],
                residual = estimate['mean_reproj_error']
            )
            scale_summary_by_cam.setdefault(cam_id, {}).update({
                'joint_scale': float(estimate['scale']),
                'joint_scale_points': int(estimate['points']),
                'joint_scale_applied': bool(updated),
            })
            for pair_name, data in estimate.get('pair_means', {}).items():
                update_scale_pair_estimate(
                    scale_summary_by_cam, cam_id, pair_name, pair_side_lookup,
                    data.get('mean_scale', np.nan), data.get('points', 0)
                )
            run_logger.log_scale_points(trackers[cam_id], estimate, updated, pair_side_lookup)
            if updated:
                pair_info = ', '.join(
                    f"{pair}: {data['mean_scale']:.6g} ({data['points']})"
                    for pair, data in estimate['pair_means'].items()
                )
                print(
                    f"Cam {cam_id}: metric scale = {estimate['scale']:.6g} m/unit, "
                    f"points = {estimate['points']}, "
                    f"reproj = {estimate['mean_reproj_error']:.2f}px, "
                    f"hamming = {estimate['mean_hamming']:.1f}, "
                    f"pairs = [{pair_info}], source = {obs['source']}"
                )

        run_logger.log_keyframe_events(trackers)
        run_logger.log_motion_points(trackers)
        run_logger.log_frame_motion(trackers)
        run_logger.log_scale_summary(trackers, scale_summary_by_cam)

        if DISPLAY_CONFIG.get('show_cross_camera_points', True):
            dedupe_radius = DISPLAY_CONFIG.get('cross_visual_dedupe_radius_px', 5.0)
            for cam_id, observations in cross_visual_observations.items():
                if 0 <= cam_id < len(frame_layers) and frame_layers[cam_id] is not None:
                    draw_cross_camera_observations_dedup(
                        frame_layers[cam_id],
                        observations,
                        radius_px = dedupe_radius
                    )

        # Вставка видео-слоев в холсты
        for i, cv in enumerate(video_canvases):
            if cv is not None and frame_layers[i] is not None:
                paste_video_layer_into_canvas(cv, frame_layers[i])
                if active[i] and trackers[i] is not None and hasattr(trackers[i], 'refresh_last_video_canvas_panel'):
                    trackers[i].last_video_canvas = cv
                    refreshed = trackers[i].refresh_last_video_canvas_panel()
                    if refreshed is not None:
                        video_canvases[i] = refreshed.astype(np.uint8)

        # Приведение всех видео к единому размеру
        video_canvases = pad_to_max_size(video_canvases)

        # Сборка сетки 2x2 из видео
        row1_video = np.hstack([video_canvases[0], video_canvases[1]])
        row2_video = np.hstack([video_canvases[2], video_canvases[3]])
        video_grid = np.vstack([row1_video, row2_video])

        # Сбор и усреднение данных о движении
        motion_states = collect_motion_states(trackers, active)
        avg_data = average_motion_by_placement(motion_states, rotations = anchor_rotations) if motion_states else None

        # Обновление счетчиков движения
        run_logger.update_local_unit_motion_counts(motion_states, avg_data)
        run_logger.log_local_unit_z_motion(motion_states, avg_data)

        # Визуализация общей панели
        summary_col = render_summary_column(
            avg_data, motion_states, trackers, active,
            height = video_grid.shape[0],
            width = DISPLAY_CONFIG['summary_col_width'],
            match_counts = match_counts,
            font_scale = DISPLAY_CONFIG['font_scale'],
            thickness = DISPLAY_CONFIG['font_thickness'],
            line_height = DISPLAY_CONFIG['line_height'],
            section_gap = DISPLAY_CONFIG['section_gap'],
            margin_x = DISPLAY_CONFIG['text_margin_x'],
            start_y = DISPLAY_CONFIG['text_y_start']
        )
        video_grid = np.hstack([video_grid, summary_col])

        # Запись финального окна в видео
        final_window_recorder.write(video_grid)

        # Сборка кадра с репроекциями ключевых кадров
        kf_reprojection_frame = build_keyframe_reprojection_video_frame(
            kf_canvases,
            grid_cols = RECORDING_CONFIG.get('keyframe_grid_cols', 2)
        )
        kf_reprojection_recorder.write(kf_reprojection_frame)

        # Отображение на экране
        cv2.imshow(DISPLAY_CONFIG['window_video'], video_grid)
        if cv2.waitKey(1) & 0xFF  == ord('q'):
            break

    final_window_recorder.release()
    kf_reprojection_recorder.release()
    run_logger.close()

    for tr in trackers:
        if tr is not None:
            tr.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
