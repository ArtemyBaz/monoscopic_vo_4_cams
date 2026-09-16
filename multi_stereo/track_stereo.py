import cv2
import numpy as np
from typing import Sequence, Tuple, Optional

from stereo_config_USB import (
    ORB_DEFAULTS, # Настройки дискриптора 
    TRACKER_DEFAULTS, # Настройки трекера
    TRACKER_DISPLAY_DEFAULTS, # Настройки визуализации
    MODEL_ESTIMATION_DEFAULTS, # Настройки гомографии и существенной матрицы
    POSE_MATCH_CONFIG, # Настройки матчера
    PNP_CONFIG, # Настройки PnP
    KEYFRAME_CONFIG, # Настройки ключевых кадров
    EVALUATE_PAIR_CONFIG, # Настройки инициализации
)


# Хранение массива характерных точек
class FeatureSet:
    def __init__(self, points: np.ndarray, descriptors: np.ndarray):
        self.points = points
        self.descriptors = descriptors

    def copy(self) -> "FeatureSet":
        return FeatureSet(
            self.points.copy(),
            self.descriptors.copy() if self.descriptors is not None else None
        )


# Создание пустого массива характерных точек
def empty_feature_set() -> FeatureSet:
    return FeatureSet(
        np.empty((0, 2), dtype = np.float32),
        np.empty((0, 32), dtype = np.uint8)
    )


# Модель калибровки камеры
class Scaramuzza_Model:
    def __init__(self, mapp_coeff: Sequence[float], img_size: Tuple[int, int],
                 dist_cnt: Tuple[float, float], stretch_matrix: Sequence[Sequence[float]]):
        self.cx = float(dist_cnt[0]) # Центр дисторсии по x
        self.cy = float(dist_cnt[1]) # Центр дисторсии по y
        self.a0 = float(mapp_coeff[0]) # Коэффициент полома a0
        self.a2 = float(mapp_coeff[1]) # Коэффициент полома a2
        self.a3 = float(mapp_coeff[2]) # Коэффициент полома a3
        self.a4 = float(mapp_coeff[3]) # Коэффициент полома a4
        S = np.array(stretch_matrix, dtype = np.float64)
        self.A = S # Матрица афинного преобразования
        self.A_inv = np.linalg.inv(S) # Обратная матрица афинного преобразования

    # Пиксель в луч
    def pixel_to_ray_vectorized(self, points: np.ndarray) -> np.ndarray:
        u_pix = points[:, 0]
        v_pix = points[:, 1]
        
        du = u_pix - self.cx
        dv = v_pix - self.cy
        
        xy = np.dot(self.A_inv, np.vstack([du, dv])).T
        x, y = xy[:, 0], xy[:, 1]
        
        rho = np.hypot(x, y)
        rho2 = rho * rho
        rho3 = rho2 * rho
        rho4 = rho3 * rho
        z = self.a0 + self.a2 * rho2 + self.a3 * rho3 + self.a4 * rho4
        
        rays = np.stack([x, y, z], axis = 1)
        norms = np.linalg.norm(rays, axis = 1, keepdims = True)
        rays = np.divide(rays, norms, where = norms > 0)
        
        return rays

    # Нормализация 3d-точки в луч
    def point_to_ray_vectorized(self, points_3d: np.ndarray, R = None, t = None) -> np.ndarray:
        P = np.asarray(points_3d, dtype = np.float64)

        # Перенос точек в другой кадр
        if R is not None and t is not None:
            P = (R @ P.T).T + t.reshape(1, 3)
            
        norms = np.linalg.norm(P, axis = 1, keepdims = True)
        rays = np.divide(P, norms, where = norms > 1e-12)
        
        return rays

    # Проекция лучей на плоскость
    def ray_to_plane_vectorized(self, rays: np.ndarray) -> np.ndarray:
        X = rays[:, 0]
        Y = rays[:, 1]
        Z = rays[:, 2]

        # Деление с проверкой на ненулевцю Z
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            u = np.divide(X, Z, where = np.abs(Z) > 1e-8, out = np.full_like(X, 1e6))
            v = np.divide(Y, Z, where = np.abs(Z) > 1e-8, out = np.full_like(Y, 1e6))
            
        return np.column_stack((u, v))

    # Луч в пиксель
    def ray_to_pixel_vectorized(self, rays: np.ndarray) -> np.ndarray:
        X = rays[:, 0]
        Y = rays[:, 1]
        Z = rays[:, 2]
        R = np.hypot(X, Y)
        
        mask_center = R < 1e-12 # Маска лучей в центр
        
        alpha = np.divide(Z, R, where = ~mask_center, out = np.zeros_like(Z))
        
        rho = np.maximum(alpha, 0.0)
        rho[rho == 0] = 1.0
        
        a0, a2, a3, a4 = self.a0, self.a2, self.a3, self.a4
        # Решение полинома методом Ньютона
        for _ in range(5):
            rho2 = rho * rho
            rho3 = rho2 * rho
            rho4 = rho3 * rho
            f = a4 * rho4 + a3 * rho3 + a2 * rho2 - alpha * rho + a0
            df = 4.0 * a4 * rho3 + 3.0 * a3 * rho2 + 2.0 * a2 * rho - alpha
            mask = np.abs(df) > 1e-12
            delta = np.zeros_like(rho)
            delta[mask] = f[mask] / df[mask]
            rho -= delta
            if np.max(np.abs(delta)) < 1e-8:
                break
            
        target_rho = rho
        target_rho[mask_center] = 0.0
        
        u = X / R * target_rho
        v = Y / R * target_rho
        
        du = self.A[0,0] * u + self.A[0,1] * v
        dv = self.A[1,0] * u + self.A[1,1] * v
        u_pix = self.cx + du
        v_pix = self.cy + dv
        u_pix[mask_center] = self.cx
        v_pix[mask_center] = self.cy
        
        return np.column_stack((u_pix, v_pix))


# Поиск характерных точек и дескрипторов
class Orb_search:
    def __init__(self, nfeatures = ORB_DEFAULTS['nfeatures'], # Максимальное количество точек
                 scaleFactor = ORB_DEFAULTS['scaleFactor'], # Коэффициент масшаба между уровнями пирамиды
                 nlevels = ORB_DEFAULTS['nlevels']): # Количество уровней в пирамиде
        score_type = ORB_DEFAULTS.get('scoreType', 'HARRIS')
        
        if isinstance(score_type, str):
            score_type = cv2.ORB_FAST_SCORE if score_type.upper() == 'FAST' else cv2.ORB_HARRIS_SCORE

        self.orb = cv2.ORB_create(
            nfeatures = nfeatures,
            scaleFactor = scaleFactor,
            nlevels = nlevels,
            edgeThreshold = ORB_DEFAULTS.get('edgeThreshold', 31), # Края изображения, на которых точки не ищутся
            firstLevel = ORB_DEFAULTS.get('firstLevel', 0),
            WTA_K = ORB_DEFAULTS.get('WTA_K', 2), # Дескриптор BRIEF
            scoreType = score_type,
            patchSize = ORB_DEFAULTS.get('patchSize', 31), # Область для вычисления дескриптора
            fastThreshold = ORB_DEFAULTS.get('fastThreshold', 20) # Порог детектора
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) # Матчер по расстоянию Хэмминга

    # Поиск характерных точек
    def extract(self, frame) -> FeatureSet:
        if frame is None:
            return empty_feature_set()
        
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        kp, descriptors = self.orb.detectAndCompute(gray, None)
        
        if not kp or descriptors is None or len(descriptors) == 0:
            return empty_feature_set()
        
        points = np.array([p.pt for p in kp], dtype = np.float32)
        descriptors = np.asarray(descriptors, dtype = np.uint8)
        
        return FeatureSet(points, descriptors)

    # Точки и дескрипторы для 2 кадров
    def match_feature_sets(self, features1: FeatureSet, features2: FeatureSet, max_distance: Optional[float] = None):
        return self.match_features(
            features1.points, features1.descriptors,
            features2.points, features2.descriptors,
            max_distance = max_distance
        )

    # Сопоставление точек
    def match_features(self, key_p1, descrip1, key_p2, descrip2, max_distance: Optional[float] = None):
        key_p1 = np.asarray(key_p1, dtype = np.float32).reshape(-1, 2)
        key_p2 = np.asarray(key_p2, dtype = np.float32).reshape(-1, 2)
        empty_pts = np.empty((0, 2), dtype = np.float32)
        
        if (descrip1 is None or descrip2 is None or
                len(key_p1) == 0 or len(key_p2) == 0 or
                len(descrip1) == 0 or len(descrip2) == 0):
            return empty_pts, empty_pts, []
        
        descrip1 = np.asarray(descrip1, dtype = np.uint8)
        descrip2 = np.asarray(descrip2, dtype = np.uint8)

        # Сопоставление по дескрипторам
        matches = self.bf.match(descrip1, descrip2)

        # Фильтрация по расстоянию Хэмминга 
        if max_distance is not None:
            matches = [m for m in matches if m.distance <= max_distance]
  
        matches = sorted(matches, key = lambda x: x.distance)
        
        if not matches:
            return empty_pts, empty_pts, []
        
        query_idx = np.fromiter((m.queryIdx for m in matches), dtype = np.int32, count = len(matches))
        train_idx = np.fromiter((m.trainIdx for m in matches), dtype = np.int32, count = len(matches))
        pts1_matched = key_p1[query_idx].astype(np.float32, copy = False)
        pts2_matched = key_p2[train_idx].astype(np.float32, copy = False)
        
        return pts1_matched, pts2_matched, matches


# Триангуляция точек
def triangulate_points(pts1_uv: np.ndarray,
                       pts2_uv: np.ndarray,
                       R: np.ndarray, # Матрица поворота
                       t: np.ndarray, # Матрица переноса
                       K: np.ndarray = np.eye(3)) -> np.ndarray:
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))
    
    pts4d = cv2.triangulatePoints(P1, P2, pts1_uv.T, pts2_uv.T)
    
    pts4d /= (pts4d[3] + 1e-8)
    pts3d = pts4d[:3].T
    
    return pts3d


# Отрисовка репроецированных точек
# Зелёные точки - 2d
# Синие точки - 3d
# Красные линии - соответствие точек
# Число - глубина точки
def draw_triangulated_points(frame,
                             pts_pix,
                             depths,
                             reproj_pixels = None,
                             reproj_color = (255, 0, 0),
                             max_points_to_draw = None
                             ):
    img_out = frame.copy()
    
    if len(pts_pix) == 0:
        cv2.putText(img_out, "No points", (30, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        return img_out

    # Фильтрация точек с положительной глубиной
    valid = depths > 0
    pts_valid = pts_pix[valid]
    depths_valid = depths[valid]
    
    if len(pts_valid) == 0:
        cv2.putText(img_out, "No points with +Z", (30, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        return img_out
    
    reproj_valid = None
    if reproj_pixels is not None:
        if len(reproj_pixels) == len(pts_pix):
            reproj_valid = reproj_pixels[valid]
        else:
            reproj_valid = reproj_pixels

    # Ограниечение количества точек для отображения        
    N = len(pts_valid)
    if max_points_to_draw is not None and N > max_points_to_draw:
        indices = np.random.choice(N, max_points_to_draw, replace = False)
        cv2.putText(img_out, f"Show {max_points_to_draw} of {N} points", (30, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    else:
        indices = np.arange(N)

    # Отрисовка точек    
    for idx in indices:
        u, v = pts_valid[idx].astype(int)
        depth = depths_valid[idx]
        
        in_bounds_orig = (0 <= u < frame.shape[1] and 0 <= v < frame.shape[0])
        if in_bounds_orig:
            cv2.circle(img_out, (u, v), 3, (0, 255, 0), -1)
            text = f"{depth:.1f}"
            cv2.putText(img_out, text, (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        if reproj_valid is not None and idx < len(reproj_valid):
            ru, rv = reproj_valid[idx]
            ru_int, rv_int = int(round(ru)), int(round(rv))
            in_bounds_reproj = (0 <= ru_int < frame.shape[1] and 0 <= rv_int < frame.shape[0])
            if in_bounds_reproj:
                cv2.circle(img_out, (ru_int, rv_int), 2, reproj_color, -1)
            if in_bounds_orig and in_bounds_reproj:
                cv2.line(img_out, (u, v), (ru_int, rv_int), (0, 0, 255), 1)
                
    return img_out


# Отрисовка репроецированных точек на текущий ключевой кадр
# Зелёные точки - 2d
# Жёлтые точки - 3d
# Красные линии - соответствие точек
# Число - глубина точки
def render_keyframe_reprojection(camera, frame, pts2d, pts3d, kf_id, additional_info = "", R = None, t = None,
                                 panel_width = 220, font_scale = 0.5, thickness = 1,
                                 line_height = 20, margin_x = 10, start_y = 20):
    if R is not None and t is not None:
        rays = camera.point_to_ray_vectorized(pts3d, R = R, t = t)
    else:
        rays = camera.point_to_ray_vectorized(pts3d)
    reproj_pixels = camera.ray_to_pixel_vectorized(rays)
    img_with_points = draw_triangulated_points(frame, pts2d, pts3d[:, 2], reproj_pixels, reproj_color = (0, 255, 255))
    h, w = frame.shape[:2]
    panel_width = max(0, int(panel_width))
    canvas = np.ones((h, w + panel_width, 3), dtype = np.uint8) * 255
    canvas[:, panel_width:] = img_with_points
    y_offset = int(start_y)

    panel = canvas[:, :panel_width] if panel_width > 0 else None

    # Вывод текста на панель
    def put(text):
        nonlocal y_offset
        if panel is not None and y_offset <= h - 8:
            cv2.putText(panel, text, (int(margin_x), y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), (0, 0, 0), int(thickness))
        y_offset += int(line_height)

    put(f"Keyframe {kf_id}")
    put(f"Points: {len(pts2d)}")
    if additional_info:
        y_offset += max(0, int(line_height // 2))
        for line in additional_info.split('\n'):
            put(line)
            
    return canvas


# Перевод матрицы поворота в углы Эйлера
def rotation_matrix_to_euler(R: np.ndarray, degrees: bool = True) -> Tuple[float, float, float]:
    sy = np.sqrt(R[0,0] ** 2 + R[1,0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        rz = np.arctan2(R[1,0], R[0,0])
        ry = np.arctan2(-R[2,0], sy)
        rx = np.arctan2(R[2,1], R[2,2])
    else:
        rz = np.arctan2(-R[1,2], R[1,1])
        ry = np.arctan2(-R[2,0], sy)
        rx = 0.0 # при 90 градусов фиксируем поворот вокруг оси x
    if degrees:
        rx = np.degrees(rx)
        ry = np.degrees(ry)
        rz = np.degrees(rz)
        
    return rx, ry, rz


# Приведение уголов к диапазону 0 - 360 градусов
def normalize_angles_360(angles: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(angles, dtype = np.float64), 360.0)


# Приведение уголов к диапазону -180 - 180 градусов
def signed_delta_angles_deg(angles_deg: np.ndarray) -> np.ndarray:
    return (np.asarray(angles_deg, dtype = np.float64) + 180.0) % 360.0 - 180.0


# Вычисление перемещения между кадрами
def pnp_translation_to_camera_delta(R_rel: np.ndarray, t_rel: np.ndarray) -> np.ndarray:
    R_rel = np.asarray(R_rel, dtype = np.float64).reshape(3, 3)
    t_rel = np.asarray(t_rel, dtype = np.float64).reshape(3)
    
    return -R_rel.T @ t_rel


# Вычисление позиции кадра в глобальной системе
def extrinsic_to_camera_center(R_cw: np.ndarray, t_cw: np.ndarray) -> np.ndarray:
    R_cw = np.asarray(R_cw, dtype = np.float64).reshape(3, 3)
    t_cw = np.asarray(t_cw, dtype = np.float64).reshape(3)
    
    return -R_cw.T @ t_cw


# Нормализация истории точек
def normalize_pixel_history(history, max_len: int):
    max_len = max(0, int(max_len))
    
    if max_len <= 0 or history is None:
        return []
    
    arr = np.asarray(history, dtype = np.float64)
    if arr.size == 0:
        return []
    
    arr = arr.reshape(-1, 2)
    finite = np.isfinite(arr).all(axis = 1)
    out = [p.astype(np.float64, copy = True) for p in arr[finite]]
    
    return out[-max_len:]


# Создание истории точек
def make_point_follow_histories(pixels, max_prev_frames: int):
    max_prev_frames = max(0, int(max_prev_frames))
    pts = np.asarray(pixels, dtype = np.float64).reshape(-1, 2) if pixels is not None else np.empty((0, 2))
    
    if max_prev_frames <= 0:
        return [[] for _ in range(len(pts))]
    
    histories = []
    for p in pts:
        histories.append([p.astype(np.float64, copy = True)] if np.isfinite(p).all() else [])
        
    return histories


# Накопление истории точек
def extend_point_follow_history(history, current_point, max_prev_frames: int):
    out = normalize_pixel_history(history, max_prev_frames)
    max_prev_frames = max(0, int(max_prev_frames))
    
    if max_prev_frames <= 0:
        return []
    
    cur = np.asarray(current_point, dtype = np.float64).reshape(2)
    if np.isfinite(cur).all():
        out.append(cur.astype(np.float64, copy = True))
        
    return out[-max_prev_frames:]


# Оценка естественности движения точек
def point_follow_motion_is_natural(history, current_point,
                                   prev_frames: int,
                                   max_prediction_error_px: float,
                                   prediction_error_factor: float,
                                   max_step_px: float,
                                   step_factor: float,
                                   min_motion_px: float) -> bool:
    prev_frames = max(0, int(prev_frames))
    if prev_frames <= 0:
        return True

    hist = normalize_pixel_history(history, prev_frames)
    if len(hist) < prev_frames:
        return True

    cur = np.asarray(current_point, dtype = np.float64).reshape(2)
    if not np.isfinite(cur).all():
        return False

    pts = np.vstack(hist[-prev_frames:] + [cur])
    if not np.isfinite(pts).all():
        return False

    # Последний шаг
    last_vec = pts[-1] - pts[-2]
    last_step = float(np.linalg.norm(last_vec))

    # Вычисление предыдущих шагов
    if prev_frames >= 2:
        prev_vecs = np.diff(pts[:-1], axis = 0)
        prev_lengths = np.linalg.norm(prev_vecs, axis = 1)
        finite_prev = np.isfinite(prev_lengths)
        prev_vecs = prev_vecs[finite_prev]
        prev_lengths = prev_lengths[finite_prev]
    else:
        prev_vecs = np.empty((0, 2), dtype = np.float64)
        prev_lengths = np.empty(0, dtype = np.float64)

    # Медиана по шагам
    typical_step = float(np.median(prev_lengths)) if len(prev_lengths) else 0.0
    motion_floor = max(float(min_motion_px), 1e-6)

    # Проверка текцщего шага
    step_limit = max(float(max_step_px), float(step_factor) * max(typical_step, motion_floor))
    if last_step > step_limit:
        return False

    # Проверка траектории
    if len(prev_vecs):
        expected_vec = np.median(prev_vecs, axis = 0)
        predicted = pts[-2] + expected_vec
        prediction_error = float(np.linalg.norm(cur - predicted))
        prediction_limit = max(
            float(max_prediction_error_px),
            float(prediction_error_factor) * max(typical_step, motion_floor)
        )
        if prediction_error > prediction_limit:
            return False

    return True


# Вычисление гомографии
class Search_homography:
    def __init__(self, ransac_threshold = MODEL_ESTIMATION_DEFAULTS['homography_ransac_threshold']):
        self.ransac_threshold = ransac_threshold
        self.H = None
        self.mask = None
        self.nb_solutions = 0
        self.Rs = []
        self.Ts = []
        self.Ns = []
        self.best_R = None
        self.best_t = None
        self.best_n = None
        self.score = 0.0

    # Поиск матрицы гомографии
    def compute(self, pts1_uv: np.ndarray, pts2_uv: np.ndarray,
                frame1 = None, frame2 = None,
                pts1_pix = None, pts2_pix = None) -> bool:
        H, mask = cv2.findHomography(pts1_uv, pts2_uv, cv2.RANSAC, self.ransac_threshold)
        if H is None:
            return False
        
        self.H = H
        if mask is not None:
            self.mask = mask.ravel().astype(bool)
        else:
            self.mask = np.ones(len(pts1_uv), dtype = bool)
            
        self.score = self._compute_score(pts1_uv, pts2_uv, H, np.linalg.inv(H)) # Вычисление качества гомографии

        # Вычисление поворота, перемещения и нормали плоскости из гомографии
        K_eye = np.eye(3, dtype = np.float32)
        self.nb_solutions, self.Rs, self.Ts, self.Ns = cv2.decomposeHomographyMat(H, K_eye)
        best_idx = 0

        # Выбор решения из 2/4 по нормали плоскости
        for i, n in enumerate(self.Ns):
            if n[2] > 0:
                best_idx = i
                break
        self.best_R = self.Rs[best_idx]
        self.best_t = self.Ts[best_idx]
        self.best_n = self.Ns[best_idx]
        
        if frame1 is not None and frame2 is not None and pts1_pix is not None and pts2_pix is not None and self.mask is not None:
            draw_matches(frame1, pts1_pix, frame2, pts2_pix, self.mask, "Гомография")
            
        return True

    # Вычисление качества гомографии
    def _compute_score(self, pts1, pts2, H21, H12):
        th = 5.991 # Порог как в ORB-SLAM3 (доверительный интервал 95%)
        inv_sigma2 = 1.0
        
        pts1 = np.asarray(pts1, dtype = np.float64)
        pts2 = np.asarray(pts2, dtype = np.float64)
        u1, v1 = pts1[:, 0], pts1[:, 1]
        u2, v2 = pts2[:, 0], pts2[:, 1]
        
        h11, h12, h13 = H21[0,0], H21[0,1], H21[0,2]
        h21, h22, h23 = H21[1,0], H21[1,1], H21[1,2]
        h31, h32, h33 = H21[2,0], H21[2,1], H21[2,2]
        h11i, h12i, h13i = H12[0,0], H12[0,1], H12[0,2]
        h21i, h22i, h23i = H12[1,0], H12[1,1], H12[1,2]
        h31i, h32i, h33i = H12[2,0], H12[2,1], H12[2,2]
        
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            # Репроекция точек из кадра 2 в кадр 1
            w2in1 = 1.0 / (h31i * u2 + h32i * v2 + h33i)
            u2in1 = (h11i * u2 + h12i * v2 + h13i) * w2in1
            v2in1 = (h21i * u2 + h22i * v2 + h23i) * w2in1
            
            # Ошибка репроекции
            err1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1)
            chi1 = err1 * inv_sigma2

            # Репроекция точек из кадра 1 в кадр 2
            w1in2 = 1.0 / (h31 * u1 + h32 * v1 + h33)
            u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2
            v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2

            # Ошибка репроекции
            err2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2)
            chi2 = err2 * inv_sigma2

        # Вычисление score по хорошим точкам
        valid1 = np.isfinite(chi1) & (chi1 < th)
        valid2 = np.isfinite(chi2) & (chi2 < th)
        score = np.sum(th - chi1[valid1]) + np.sum(th - chi2[valid2])
        
        return float(score)

    # Отображение результатов
    def print_results(self):
        if self.H is None:
            print("Гомография не найдена")
            return
        print("\nМатрица гомографии H:")
        print(self.H)
        print(f"Найдено решений: {self.nb_solutions}")
        for i in range(self.nb_solutions):
            print(f"\n--- Решение гомографии {i+1} ---")
            print("R:\n", self.Rs[i])
            print("t:", self.Ts[i].ravel())
            print("n:", self.Ns[i].ravel())
        print(f"\nВыбранное решение из {self.nb_solutions} (гомография):")
        print("R:\n", self.best_R)
        print("t:", self.best_t.ravel())
        print("n:", self.best_n.ravel())

    def get_score(self):
        return self.score


# Вычисление существенной матрицы
class Search_essential_matrix:
    def __init__(self, ransac_threshold = MODEL_ESTIMATION_DEFAULTS['essential_ransac_threshold'],
                 prob = MODEL_ESTIMATION_DEFAULTS['essential_prob']):
        self.ransac_threshold = ransac_threshold
        self.prob = prob
        self.E = None
        self.mask = None
        self.all_solutions = []
        self.best_R = None
        self.best_t = None
        self.valid_mask = None
        self.score = 0.0

    def compute(self, pts1_uv: np.ndarray, pts2_uv: np.ndarray,
                frame1 = None, frame2 = None,
                pts1_pix = None, pts2_pix = None) -> bool:
        E, mask = cv2.findEssentialMat(pts1_uv, pts2_uv,
                                       cameraMatrix = np.eye(3),
                                       method = cv2.RANSAC,
                                       prob = self.prob,
                                       threshold = self.ransac_threshold)
        if E is None:
            return False
        self.E = E
        if mask is not None:
            self.mask = mask.ravel().astype(bool)
        else:
            self.mask = np.ones(len(pts1_uv), dtype = bool)
        self.score = self._compute_score(pts1_uv, pts2_uv, E)

        # 4 решения поворота и переноса существенной матрицы
        R1, R2, t = cv2.decomposeEssentialMat(E)
        self.all_solutions = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
        self._choose_best_solution(pts1_uv, pts2_uv) # Выбор решения
        
        if frame1 is not None and frame2 is not None and pts1_pix is not None and pts2_pix is not None and self.mask is not None:
            draw_matches(frame1, pts1_pix, frame2, pts2_pix, self.mask, "Существенная матрица")

        return True

    def _compute_score(self, pts1, pts2, F):
        th = 3.841 # Порог как в ORB-SLAM3 (доверительный интервал 95%)
        thScore = 5.991
        inv_sigma2 = 1.0
        
        pts1 = np.asarray(pts1, dtype = np.float64)
        pts2 = np.asarray(pts2, dtype = np.float64)
        u1, v1 = pts1[:, 0], pts1[:, 1]
        u2, v2 = pts2[:, 0], pts2[:, 1]

        # Разложение матрицы
        f11, f12, f13 = F[0,0], F[0,1], F[0,2]
        f21, f22, f23 = F[1,0], F[1,1], F[1,2]
        f31, f32, f33 = F[2,0], F[2,1], F[2,2]

        # Вычисление эпиполярной ошибки
        a2 = f11 * u1 + f12 * v1 + f13
        b2 = f21 * u1 + f22 * v1 + f23
        c2 = f31 * u1 + f32 * v1 + f33
        num2 = a2 * u2 + b2 * v2 + c2
        a1 = f11 * u2 + f21 * v2 + f31
        b1 = f12 * u2 + f22 * v2 + f32
        c1 = f13 * u2 + f23 * v2 + f33
        num1 = a1 * u1 + b1 * v1 + c1

        # Вычисление score по хорошим точкам        
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            dist2 = num2 * num2 / (a2 * a2 + b2 * b2)
            chi2 = dist2 * inv_sigma2
            dist1 = num1 * num1 / (a1 * a1 + b1 * b1)
            chi1 = dist1 * inv_sigma2
        valid2 = np.isfinite(chi2) & (chi2 < th)
        valid1 = np.isfinite(chi1) & (chi1 < th)
        score = np.sum(thScore - chi2[valid2]) + np.sum(thScore - chi1[valid1])
        
        return float(score)

    # Выбор из 4 решений по глубине
    def _choose_best_solution(self, pts1_uv, pts2_uv, K = np.eye(3)):
        best_count = -1
        best_Rt = None
        best_mask = None
        
        for R, t_vec in self.all_solutions:
            pts3d = triangulate_points(pts1_uv, pts2_uv, R, t_vec, K)
            depths2 = (R @ pts3d.T + t_vec.reshape(3,1))[2]
            valid = (pts3d[:, 2] > 0) & (depths2 > 0)
            count = np.sum(valid)
            if count > best_count:
                best_count = count
                best_Rt = (R, t_vec)
                best_mask = valid
        if best_Rt is not None:
            self.best_R, self.best_t = best_Rt
            self.valid_mask = best_mask

    # Отображение результатов
    def print_results(self):
        if self.E is None:
            print("Существенная матрица не найдена")
            return
        print("\nСущественная матрица E:")
        print(self.E)
        print("\nВсе 4 решения из E:")
        for i, (R, t_vec) in enumerate(self.all_solutions):
            print(f"\nРешение {i+1}:")
            print("R:\n", R)
            print("t:", t_vec.ravel())
        if self.best_R is not None:
            print("\nВыбранное решение из 4 (существенная матрица):")
            print("R:\n", self.best_R)
            print("t:", self.best_t.ravel())
            print(f"Точек с положительной глубиной: {np.sum(self.valid_mask)} из {len(self.valid_mask)}")
        else:
            print("Не удалось выбрать решение для E")

    def get_score(self):
        return self.score


# Визуализация совпадений между кдрами
def draw_matches(frame1, pts1_pix, frame2, pts2_pix, mask, window_name):
    combo = cv2.hconcat([frame1, frame2])
    w = frame1.shape[1]
    
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts1_pix, pts2_pix)):
        color = (0, 255, 0) if mask[i] else (0, 0, 255)
        cv2.line(combo, (int(x1), int(y1)), (int(x2 + w), int(y2)), color, 1)
        cv2.circle(combo, (int(x1), int(y1)), 4, color, 2)
        cv2.circle(combo, (int(x2 + w), int(y2)), 4, color, 2)
        
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, combo)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


# Выбор между гомографией и существенной матрицей
def select_best_model(homo, ess, uv1, uv2):
    ok_h = homo.H is not None
    ok_e = ess.E is not None
    
    score_h = homo.get_score() if ok_h else 0.0
    score_e = ess.get_score() if ok_e else 0.0
    
    if ok_h:
        homo.print_results()
    if ok_e:
        ess.print_results()
    print(f"\nScore гомографии: {score_h:.2f}")
    print(f"Score существенной матрицы: {score_e:.2f}")
    
    if not ok_h and not ok_e:
        print("Гомография и существенная матрица не вычислены")
        return None
    
    if ok_h and ok_e:
        total = score_h + score_e
        rh = score_h / total # Отношение score гомографии к сумме score
        if rh > 0.5:
            R, t = homo.best_R, homo.best_t
            pts1, pts2 = uv1, uv2
            mask = homo.mask
            score = score_h
            print(f"\nВыбрана гомография: (RH = {rh:.3f} > 0.5)")
        else:
            R, t = ess.best_R, ess.best_t
            pts1, pts2 = uv1, uv2
            mask = ess.mask
            score = score_e
            print(f"\nВыбрана существенная матрица: (RH = {rh:.3f} <= 0.5)")
    elif ok_h:
        R, t = homo.best_R, homo.best_t
        pts1, pts2 = uv1, uv2
        mask = homo.mask
        score = score_h
        rh = 1.0
        print(f"\nВыбрана гомография матрица, так как существенной матрицы нет")
    else:
        R, t = ess.best_R, ess.best_t
        pts1, pts2 = uv1, uv2
        mask = ess.mask
        score = score_e
        rh = 0.0
        print(f"\nВыбрана существенная матрица, так как гомографии нет")
        
    return {
        'R': R,
        't': t,
        'pts1_uv': pts1,
        'pts2_uv': pts2,
        'mask': mask,
        'score': score,
        'rh': rh
    }


# Визуализация инициализации системы
def render_init_system(camera, orb, frame_prev, frame_curr, pts1_pix, pts2_pix,
                       matches, descrip_prev, descrip_curr,
                       homo_ransac_threshold, ess_ransac_threshold, ess_prob,
                       show_model_matches = True,
                       features_prev: Optional[FeatureSet] = None,
                       features_curr: Optional[FeatureSet] = None):
    if descrip_prev is None:
        if features_prev is None:
            features_prev = orb.extract(frame_prev)
        descrip_prev = features_prev.descriptors
        
    if descrip_curr is None:
        if features_curr is None:
            features_curr = orb.extract(frame_curr)
        descrip_curr = features_curr.descriptors

    # Извлечение дескрипторов   
    query_idx = np.fromiter((m.queryIdx for m in matches), dtype = np.int32, count = len(matches))
    train_idx = np.fromiter((m.trainIdx for m in matches), dtype = np.int32, count = len(matches))
    desc1_matched = descrip_prev[query_idx]
    desc2_matched = descrip_curr[train_idx]

    # Пиксели в лучи
    rays1 = camera.pixel_to_ray_vectorized(pts1_pix)
    uv1 = camera.ray_to_plane_vectorized(rays1)
    rays2 = camera.pixel_to_ray_vectorized(pts2_pix)
    uv2 = camera.ray_to_plane_vectorized(rays2)

    # Вычисление гомографии и существенной матрицы
    homo = Search_homography(ransac_threshold = homo_ransac_threshold)
    ess = Search_essential_matrix(ransac_threshold = ess_ransac_threshold, prob = ess_prob)
    if show_model_matches:
        homo.compute(uv1, uv2, frame_prev, frame_curr, pts1_pix, pts2_pix)
        ess.compute(uv1, uv2, frame_prev, frame_curr, pts1_pix, pts2_pix)
    else:
        homo.compute(uv1, uv2)
        ess.compute(uv1, uv2)

    # Выбор мед\жду гомографией и существенной матрицей
    best = select_best_model(homo, ess, uv1, uv2)    
    if best is None:
        return None, None

    # Фильтрация
    mask = best['mask']
    pts1_uv_inliers = best['pts1_uv'][mask]
    pts2_uv_inliers = best['pts2_uv'][mask]
    pts1_pix_inliers = pts1_pix[mask]
    pts2_pix_inliers = pts2_pix[mask]
    desc1_inliers = desc1_matched[mask]
    desc2_inliers = desc2_matched[mask]
    pts3d_cam1 = triangulate_points(pts1_uv_inliers, pts2_uv_inliers, best['R'], best['t'], K = np.eye(3))
    pts3d_cam2 = (best['R'] @ pts3d_cam1.T).T + best['t'].reshape(1, 3)
    valid_depth = (pts3d_cam1[:, 2] > 0) & (pts3d_cam2[:, 2] > 0)
    final_pts3d_cam1 = pts3d_cam1[valid_depth]
    final_pts3d_cam2 = pts3d_cam2[valid_depth]
    
    data = {
        'pts3d': final_pts3d_cam1,
        'pts3d_cam2': final_pts3d_cam2,
        'descriptors1': desc1_inliers[valid_depth],
        'descriptors2': desc2_inliers[valid_depth],
        'depths': final_pts3d_cam2[:, 2],
        'R': best['R'],
        't': best['t'].flatten(),
        'pts1_pix': pts1_pix_inliers[valid_depth],
        'pts2_pix': pts2_pix_inliers[valid_depth],
        'valid_mask': valid_depth,
        'frame_prev': frame_prev,
        'frame_curr': frame_curr
    }
    canvas = render_keyframe_reprojection(camera, frame_curr, data['pts2_pix'], final_pts3d_cam2, kf_id = 0, additional_info = "Initialization")

    return data, canvas


# Отслеживание перемещений и инициализация
class CameraTracker:
    def __init__(self, config: dict, cam_id: int, show_tracking_points: bool = True):
        config = {**TRACKER_DEFAULTS, **config}
        self.cam_id = cam_id
        self.placement = str(config['placement']).lower()
        self.cap = cv2.VideoCapture(config['video_path'])
        if not self.cap.isOpened():
            raise IOError(f"Camera {cam_id}: cannot open video {config['video_path']}")
        self.camera = Scaramuzza_Model(config['mapp_coeff'], config['img_size'],
                                       config['dist_cnt'], config['stretch_matrix'])
        self.orb = Orb_search(nfeatures = config['nfeatures'],
                              scaleFactor = config['scaleFactor'],
                              nlevels = config['nlevels'])
        
        # Параметры инициализации
        self.score_threshold = config['score_threshold']
        self.min_parallax_deg = config['min_parallax_deg']
        self.min_matches_init = config['min_matches_init']
        self.min_inliers_init = config['min_inliers_init']
        self.normalize_init_scale = config['normalize_init_scale']
        self.homo_ransac_threshold = config['homo_ransac_threshold']
        self.ess_ransac_threshold = config['ess_ransac_threshold']
        self.ess_prob = config['ess_prob']

        # Параметры отслеживания перемещений
        self.search_radius = config['search_radius']
        self.hamming_threshold = config['hamming_threshold']
        self.min_matches_kf = config['min_matches_kf']
        self.kf_min_matches = config['kf_min_matches']
        self.kf_max_reproj_error = config['kf_max_reproj_error']

        self.state = 'INIT' # текущий режим
        self.first_frame = None
        self.first_features = None
        self.frame_idx = 0
        self.current_features = empty_feature_set()
        
        # Данные инициализации
        self.pending_init_result = None
        self.pending_init_frame = None
        self.pending_init_features = None
        self.pending_init_ready = False
        self.last_init_metrics = None

        # Данные отслеживания перемещений
        self.current_kf = None
        self.new_pts3d = None
        self.new_prev_descriptors = None
        self.new_rel_R = None
        self.new_rel_t = None
        self.new_Glob_R = None
        self.new_Glob_t = None
        self.global_rot_zero_deg = None

        # Настройки системы координат
        self.global_rot_display_deg = np.zeros(3, dtype = np.float64)
        self.R_cam_to_anchor = np.asarray(config.get('R_cam_to_anchor', np.eye(3)), dtype = np.float64).reshape(3, 3)
        self.kf_counter = 0
        self.show_tracking_points = show_tracking_points

        # Настройки проверки естественности движения
        self.point_follow_filter_enabled = bool(config.get('point_follow_filter_enabled', False))
        self.point_follow_prev_frames = max(0, int(config.get('point_follow_prev_frames', 3)))
        self.point_follow_max_prediction_error_px = float(config.get('point_follow_max_prediction_error_px', 25.0))
        self.point_follow_prediction_error_factor = float(config.get('point_follow_prediction_error_factor', 4.0))
        self.point_follow_max_step_px = float(config.get('point_follow_max_step_px', 80.0))
        self.point_follow_step_factor = float(config.get('point_follow_step_factor', 4.0))
        self.point_follow_min_motion_px = float(config.get('point_follow_min_motion_px', 1.0))
        self.point_follow_histories = []

        self.motion_scale = 1.0 # Приведение к масштабу при инициализации
        self.init_scale_axis = -1
        self.init_scale_axis_value = np.nan

        # Настройки метрического масштаба
        self.metric_scale = None
        self.metric_scale_source = None
        self.metric_scale_inliers = 0
        self.metric_scale_residual = np.nan
        self.metric_scale_alpha = float(config['metric_scale_alpha'])
        self.metric_scale_max_jump = float(config['metric_scale_max_jump'])

        self.last_curr_pixels = np.empty((0, 2), dtype = np.float64)
        self.last_curr_points3d_unit = np.empty((0, 3), dtype = np.float64)
        self.last_curr_descriptors = np.empty((0, 32), dtype = np.uint8)

        # Настройки логирования
        self.last_keyframe_event = None
        self.last_motion_points_for_logging = None

        self.last_motion = None
        self.pose_history = []

        self.last_video_canvas = None
        self.last_kf_canvas = None
        self.last_video_frame = None

        # Настройки отображения
        self.video_panel_width = max(0, int(config.get(
            'video_panel_width',
            config.get('panel_width', TRACKER_DISPLAY_DEFAULTS['panel_width'])
        )))
        self.kf_panel_width = max(0, int(config.get('kf_panel_width', self.video_panel_width)))
        self.display_font = cv2.FONT_HERSHEY_SIMPLEX
        self.display_font_scale = float(config.get(
            'display_font_scale',
            config.get('font_scale', TRACKER_DISPLAY_DEFAULTS['font_scale'])
        ))
        self.display_font_thickness = int(config.get(
            'display_font_thickness',
            config.get('font_thickness', TRACKER_DISPLAY_DEFAULTS['font_thickness'])
        ))
        self.display_margin_x = int(config.get('display_margin_x', TRACKER_DISPLAY_DEFAULTS['display_margin_x']))
        self.display_y_start = int(config.get('display_y_start', TRACKER_DISPLAY_DEFAULTS['display_y_start']))
        self.display_line_height = int(config.get('display_line_height', TRACKER_DISPLAY_DEFAULTS['display_line_height']))
        self.display_section_gap = int(config.get('display_section_gap', TRACKER_DISPLAY_DEFAULTS['display_section_gap']))

    # Поворот камеры относительно общей системы координат
    def set_camera_to_anchor_rotation(self, R_cam_to_anchor):
        self.R_cam_to_anchor = np.asarray(R_cam_to_anchor, dtype = np.float64).reshape(3, 3)
        self._refresh_anchor_fields_in_last_motion()
        self._refresh_metric_fields_in_last_motion()

    # Переход к общей сисстеме координат
    def _to_anchor_axes(self, vec):
        return self.R_cam_to_anchor @ np.asarray(vec, dtype = np.float64).reshape(3)

    # Обновение данных в общей сисстеме координат
    def _refresh_anchor_fields_in_last_motion(self):
        if self.last_motion is None:
            return
        
        self.last_motion['R_cam_to_anchor'] = self.R_cam_to_anchor.copy()
        for base_key, anchor_key in [
            ('local_t_unit', 'local_t_anchor_unit'),
            ('global_t_unit', 'global_t_anchor_unit'),
        ]:
            value = self.last_motion.get(base_key)
            if value is None:
                self.last_motion[anchor_key] = np.full(3, np.nan, dtype = np.float64)
            else:
                arr = np.asarray(value, dtype = np.float64).reshape(3)
                if np.isfinite(arr).all():
                    self.last_motion[anchor_key] = self._to_anchor_axes(arr)
                else:
                    self.last_motion[anchor_key] = np.full(3, np.nan, dtype = np.float64)

    # Кадр из видео
    def read_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_idx += 1
            
        return ret, frame

    # Кадр + панель с данными
    def _make_panel_canvas(self, frame, panel_width):
        self.last_video_frame = frame.copy()
        h, w = frame.shape[:2]
        panel_width = max(0, int(panel_width))
        canvas = np.ones((h, w + panel_width
                          , 3), dtype = np.uint8) * 255
        canvas[:, panel_width:] = frame
        return canvas

    # Видео с панелью
    def _make_video_canvas(self, frame, title = None):
        canvas = self._make_panel_canvas(frame, self.video_panel_width)
        if title:
            self._draw_panel_rows(canvas, [title], panel_width = self.video_panel_width)
            
        return canvas

    # Последний обработанный кадр
    def get_current_video_frame(self, copy: bool = True):
        if self.last_video_frame is None:
            return None
        return self.last_video_frame.copy() if copy else self.last_video_frame

    # Обработка некорректных значений
    @staticmethod
    def _fmt_panel_value(value, digits = 3):
        try:
            value = float(value)
        except Exception:
            return "-"
        if not np.isfinite(value):
            return "-"
        
        return f"{value:.{digits}f}"

    # Данные на панели
    def _draw_panel_rows(self, canvas, rows, panel_width = None, start_y = None):
        if canvas is None:
            return canvas
        panel_width = self.video_panel_width if panel_width is None else int(panel_width)
        panel_width = max(0, min(panel_width, canvas.shape[1]))
        if panel_width <= 0:
            return canvas
        panel = canvas[:, :panel_width]
        panel[:, :] = 255
        y = self.display_y_start if start_y is None else int(start_y)
        x = self.display_margin_x
        bottom_margin = 8
        for row in rows:
            if row is None:
                y += self.display_section_gap
                continue
            if isinstance(row, tuple):
                text, color = row
            else:
                text, color = row, (0, 0, 0)
            if y <= panel.shape[0] - bottom_margin:
                cv2.putText(
                    panel, str(text), (x, y), self.display_font,
                    self.display_font_scale, color, self.display_font_thickness
                )
            y += self.display_line_height
        return canvas

    # Заголовки панели
    def _header_rows(self, frame_idx = None):
        if frame_idx is None:
            frame_idx = self.frame_idx
        return [
            f"Cam {self.cam_id} ({self.placement})",
            f"frame {int(frame_idx)}"
        ]

    # Отрисовка панели
    def _draw_tracking_panel_from_last_motion(self, canvas):
        if canvas is None or self.last_motion is None:
            return canvas
        m = self.last_motion
        local_t = np.asarray(m.get('local_t_anchor_unit', m.get('local_t_unit', np.full(3, np.nan))), dtype = np.float64).reshape(3)
        global_t = np.asarray(m.get('global_t_anchor_unit', m.get('global_t_unit', np.full(3, np.nan))), dtype = np.float64).reshape(3)
        local_r = np.asarray(m.get('local_rot', np.full(3, np.nan)), dtype = np.float64).reshape(3)
        global_r = np.asarray(m.get('global_rot', np.full(3, np.nan)), dtype = np.float64).reshape(3)

        rows = self._header_rows(m.get('frame_idx', self.frame_idx)) + [
            "anchor axes",
            None,
            f"ULX: {self._fmt_panel_value(local_t[0])}",
            f"ULY: {self._fmt_panel_value(local_t[1])}",
            f"ULZ: {self._fmt_panel_value(local_t[2])}",
            None,
            f"UGX: {self._fmt_panel_value(global_t[0], 2)}",
            f"UGY: {self._fmt_panel_value(global_t[1], 2)}",
            f"UGZ: {self._fmt_panel_value(global_t[2], 2)}",
            None,
            f"LRX: {self._fmt_panel_value(local_r[0], 1):>5}",
            f"LRY: {self._fmt_panel_value(local_r[1], 1):>5}",
            f"LRZ: {self._fmt_panel_value(local_r[2], 1):>5}",
            None,
            f"GRX: {self._fmt_panel_value(global_r[0], 1):>5}",
            f"GRY: {self._fmt_panel_value(global_r[1], 1):>5}",
            f"GRZ: {self._fmt_panel_value(global_r[2], 1):>5}",
            None,
        ]
        if bool(m.get('metric_ready', False)):
            s = m.get('metric_scale', np.nan)
            ml = np.asarray(m.get('local_t_anchor_metric', m.get('local_t_metric', np.full(3, np.nan))), dtype = np.float64).reshape(3)
            mg = np.asarray(m.get('global_t_anchor_metric', m.get('global_t_metric', np.full(3, np.nan))), dtype = np.float64).reshape(3)
            rows += [
                f"Scale: {self._fmt_panel_value(s, 5)} m/u",
                f"MLX: {self._fmt_panel_value(ml[0])} m",
                f"MLY: {self._fmt_panel_value(ml[1])} m",
                f"MLZ: {self._fmt_panel_value(ml[2])} m",
                None,
                f"MGX: {self._fmt_panel_value(mg[0], 2)} m",
                f"MGY: {self._fmt_panel_value(mg[1], 2)} m",
                f"MGZ: {self._fmt_panel_value(mg[2], 2)} m",
            ]
        else:
            rows.append("Scale: no")
            
        return self._draw_panel_rows(canvas, rows, panel_width = self.video_panel_width)

    # Обновление панели
    def refresh_last_video_canvas_panel(self):
        if self.last_video_canvas is None:
            return None
        self._draw_tracking_panel_from_last_motion(self.last_video_canvas)
        
        return self.last_video_canvas

    # Проверка готовности инициализации
    def _is_init_result_ready(self, result):
        if result is None:
            return False
        inliers = int(result.get('inlier_count', 0))
        
        return (result['parallax_deg'] >= self.min_parallax_deg and
                result['score'] >= self.score_threshold and
                inliers >= self.min_inliers_init)

    # Проверка готовности к общей инициализации
    def is_ready_for_global_init(self):
        return self.state == 'INIT' and self.pending_init_ready and self.pending_init_result is not None

    # Визуализация результатов инициализации
    def _draw_init_result(self, canvas, result, ready, cam_id, placement, frame_idx, start_y = None,
                          show_empty_message = True):
        rows = self._header_rows(frame_idx) + [None]
        if result is None:
            if show_empty_message:
                rows.append(("Not enough matches / no model", (0, 0, 255)))
            return self._draw_panel_rows(canvas, rows, panel_width = self.video_panel_width, start_y = start_y)

        inliers = int(result.get('inlier_count', 0))
        rows += [
            f"Model: {result['model_type']}",
            f"SC: {result['score']:.1f}/{self.score_threshold:.0f}",
            f"P: {result['parallax_deg']:.2f}/{self.min_parallax_deg:.1f} deg",
            f"I: {inliers} / {self.min_inliers_init}",
            ("READY" if ready else "INIT", (0, 200, 0) if ready else (0, 0, 200)),
        ]
        
        return self._draw_panel_rows(canvas, rows, panel_width = self.video_panel_width, start_y = start_y)

    # Завершение инициализации
    def complete_initialization_from_pending(self):
        if not self.is_ready_for_global_init():
            return self.last_video_canvas, self.last_kf_canvas, "init_not_ready"
        
        frame = self.pending_init_frame
        result = self.pending_init_result
        
        data, _ = render_init_system(
            self.camera, self.orb,
            self.first_frame, frame,
            result['pts1_pix'], result['pts2_pix'],
            result['matches'], result['desc1'], result['desc2'],
            self.homo_ransac_threshold, self.ess_ransac_threshold, self.ess_prob,
            show_model_matches = False,
            features_prev = self.first_features,
            features_curr = self.pending_init_features
        )
        if data is None:
            print(f"Cam {self.cam_id}: init_system failed")
            self.pending_init_ready = False
            return self.last_video_canvas, None, "init_error"
        pts3d_in_cam2 = data['pts3d_cam2'].astype(np.float64)
        init_t = data['t'].flatten().astype(np.float64)

        # Приведение к единичному масштабу
        init_scale_axis = -1
        init_scale_axis_value = np.nan
        self.motion_scale = 1.0
        if self.normalize_init_scale and np.isfinite(init_t).all():
            abs_t = np.abs(init_t)
            init_scale_axis = int(np.argmax(abs_t)) # Определение оси с макимальным перемещением
            init_scale_axis_value = float(init_t[init_scale_axis])
            if np.isfinite(init_scale_axis_value) and abs(init_scale_axis_value) > 1e-9:
                self.motion_scale = float(1.0 / abs(init_scale_axis_value))
                pts3d_in_cam2 = pts3d_in_cam2 * self.motion_scale
                init_t = init_t * self.motion_scale
        self.init_scale_axis = init_scale_axis
        self.init_scale_axis_value = init_scale_axis_value

        # Создание первого ключевого кадра
        self.current_kf = KeyFrame(
            1, frame, data['pts2_pix'], pts3d_in_cam2,
            data['descriptors2'], data['R'], init_t,
            np.eye(3), np.zeros(3),
            feature_points = (self.pending_init_features.points if self.pending_init_features is not None else None),
            feature_descriptors = (self.pending_init_features.descriptors if self.pending_init_features is not None else None)
        )
        self.kf_counter = 1
        self.new_pts3d = pts3d_in_cam2
        self.new_prev_descriptors = data['descriptors2']
        self.new_rel_R = np.eye(3)
        self.new_rel_t = np.zeros(3)
        self.new_Glob_R = data['R']
        self.new_Glob_t = init_t

        # Сохранение точек и дескрипторов
        self.last_curr_pixels = data['pts2_pix'].astype(np.float64, copy = True)
        self.last_curr_points3d_unit = pts3d_in_cam2.astype(np.float64, copy = True)
        self.last_curr_descriptors = data['descriptors2'].astype(np.uint8, copy = True)
        self.point_follow_histories = make_point_follow_histories(
            self.last_curr_pixels, self.point_follow_prev_frames
        )

        # Глобальные повороты
        self.global_rot_zero_deg = np.array(rotation_matrix_to_euler(data['R'], degrees = True), dtype = np.float64)
        self.global_rot_display_deg = np.zeros(3, dtype = np.float64)
        self.state = 'TRACKING'
        self.pending_init_ready = False

        # Визуализация первого ключевого кадра
        self.last_kf_canvas = render_keyframe_reprojection(
            self.camera, self.current_kf.frame, self.current_kf.pts2d, self.current_kf.pts3d,
            self.current_kf.id,
            additional_info = (
                f"Sync mono init\n"
                f"axis scale: {self.motion_scale:.3g}\n"
                f"axis: {'XYZ'[init_scale_axis] if init_scale_axis >= 0 else '-'}\n"
                f"metric: pending"
            ),
            panel_width = self.kf_panel_width,
            font_scale = self.display_font_scale,
            thickness = self.display_font_thickness,
            line_height = self.display_line_height,
            margin_x = self.display_margin_x,
            start_y = self.display_y_start
        )

        # Логирование ключевого кадра
        self.last_keyframe_event = {
            'cam_id': self.cam_id,
            'placement': self.placement,
            'frame_idx': self.frame_idx,
            'keyframe_id': self.current_kf.id,
            'pts2d': self.current_kf.pts2d.astype(np.float64, copy = True),
            'pts3d_unit': self.current_kf.pts3d.astype(np.float64, copy = True),
            'descriptors': self.current_kf.descriptors.astype(np.uint8, copy = True),
            'source': 'initialization',
        }

        # Обновление данных о движении
        self.last_motion_points_for_logging = None
        self._update_last_motion(np.eye(3), np.zeros(3), data['R'], init_t, inliers = 0)
        if self.last_motion is not None:
            self.last_motion.update({
                'is_keyframe': True,
                'keyframe_id': self.current_kf.id,
                'baseline_unit': 0.0,
                'baseline_rel': np.nan,
                'min_baseline_unit': float(KEYFRAME_CONFIG.get('min_baseline_unit', np.nan)),
                'min_baseline_rel': float(KEYFRAME_CONFIG.get('min_baseline_rel', np.nan)),
                'motion_points_count': 0,
                'filtered_points_count': int(len(self.current_kf.pts3d)),
                'rotation_points_count': 0,
                'translation_points_count': 0,
            })
        self.last_video_canvas = self._make_video_canvas(frame)
        self.refresh_last_video_canvas_panel()
        
        return self.last_video_canvas, self.last_kf_canvas, "init_success"

    # Обновление данных о движении
    def _update_last_motion(self, rel_R, rel_t, glob_R, glob_t, inliers, global_rot_display = None):
        # Вычисление углов
        rx, ry, rz = rotation_matrix_to_euler(rel_R, degrees = True)
        grx_raw, gry_raw, grz_raw = rotation_matrix_to_euler(glob_R, degrees = True)
        global_rot_raw = np.array([grx_raw, gry_raw, grz_raw], dtype = np.float64)
        
        if self.global_rot_zero_deg is None:
            self.global_rot_zero_deg = global_rot_raw.copy()
        if global_rot_display is None:
            global_rot_360 = normalize_angles_360(self.global_rot_display_deg)
        else:
            global_rot_360 = normalize_angles_360(global_rot_display)
        
        rel_R = np.asarray(rel_R, dtype = np.float64).reshape(3, 3)
        glob_R = np.asarray(glob_R, dtype = np.float64).reshape(3, 3)
        local_t_extrinsic = np.asarray(rel_t, dtype = np.float64).reshape(3)
        global_t_extrinsic = np.asarray(glob_t, dtype = np.float64).reshape(3)

        # Вычисление положения камеры
        local_t_unit = pnp_translation_to_camera_delta(rel_R, local_t_extrinsic)
        global_t_unit = extrinsic_to_camera_center(glob_R, global_t_extrinsic)
        local_t_anchor_unit = self._to_anchor_axes(local_t_unit)
        global_t_anchor_unit = self._to_anchor_axes(global_t_unit)

        self.last_motion = {
            'cam_id': self.cam_id,
            'placement': self.placement,
            'frame_idx': self.frame_idx,
            'local_t': local_t_unit,
            'global_t': global_t_unit,
            'local_t_unit': local_t_unit,
            'global_t_unit': global_t_unit,
            'local_t_anchor_unit': local_t_anchor_unit,
            'global_t_anchor_unit': global_t_anchor_unit,
            'local_t_extrinsic': local_t_extrinsic,
            'global_t_extrinsic': global_t_extrinsic,
            'R_cam_to_anchor': self.R_cam_to_anchor.copy(),
            'local_rot': np.array([rx, ry, rz], dtype = np.float64),
            'global_rot': global_rot_360,
            'global_rot_raw': global_rot_raw,
            'global_rot_zero_deg': self.global_rot_zero_deg.copy(),
            'inliers': int(inliers),
            'scale': float(self.motion_scale),
            'unit_scale': float(self.motion_scale),
            'init_scale_axis': int(getattr(self, 'init_scale_axis', -1)),
            'init_scale_axis_value': float(getattr(self, 'init_scale_axis_value', np.nan)),
            'state': self.state
        }
        self._refresh_metric_fields_in_last_motion()
        
        self.pose_history.append({
            'frame_idx': self.frame_idx,
            'local_t_unit': local_t_unit.copy(),
            'global_t_unit': global_t_unit.copy(),
            'local_t_anchor_unit': local_t_anchor_unit.copy(),
            'global_t_anchor_unit': global_t_anchor_unit.copy(),
            'local_rot': self.last_motion['local_rot'].copy(),
            'global_rot': self.last_motion['global_rot'].copy(),
        })

    # Возврат данных о движении
    def get_motion_state(self):
        return self.last_motion

    # Возвращение 3d-точек
    def get_current_unit_landmarks(self):
        if self.state != 'TRACKING':
            return (np.empty((0, 2), dtype = np.float64),
                    np.empty((0, 3), dtype = np.float64))
        
        return self.last_curr_pixels.copy(), self.last_curr_points3d_unit.copy()

    # Возвращение 3d-точек с дескрипторами
    def get_current_unit_landmarks_with_descriptors(self):
        if self.state != 'TRACKING':
            return (np.empty((0, 2), dtype = np.float64),
                    np.empty((0, 3), dtype = np.float64),
                    np.empty((0, 32), dtype = np.uint8))

        n = min(len(self.last_curr_pixels), len(self.last_curr_points3d_unit), len(self.last_curr_descriptors))
        if n <= 0:
            return (np.empty((0, 2), dtype = np.float64),
                    np.empty((0, 3), dtype = np.float64),
                    np.empty((0, 32), dtype = np.uint8))

        return (
            self.last_curr_pixels[:n].copy(),
            self.last_curr_points3d_unit[:n].copy(),
            self.last_curr_descriptors[:n].copy(),
        )

    # Возвращение ORB-признаков
    def get_current_features(self, copy: bool = True) -> FeatureSet:
        if self.current_features is None:
            return empty_feature_set()
        
        return self.current_features.copy() if copy else self.current_features

    # Проверка определения метрического масштаба
    def has_metric_scale(self):
        return self.metric_scale is not None and np.isfinite(self.metric_scale) and self.metric_scale > 0

    # Обновление метрического масштаба
    def _refresh_metric_fields_in_last_motion(self):
        if self.last_motion is None:
            return
        s = float(self.metric_scale) if self.has_metric_scale() else np.nan
        self.last_motion['metric_ready'] = bool(self.has_metric_scale())
        self.last_motion['metric_scale'] = s
        self.last_motion['metric_scale_source'] = self.metric_scale_source
        self.last_motion['metric_scale_inliers'] = int(self.metric_scale_inliers)
        self.last_motion['metric_scale_residual'] = float(self.metric_scale_residual) if np.isfinite(self.metric_scale_residual) else np.nan
        self._refresh_anchor_fields_in_last_motion()

        # Вычисление метрических перемещений
        if self.has_metric_scale():
            self.last_motion['local_t_metric'] = self.last_motion['local_t_unit'] * s
            self.last_motion['global_t_metric'] = self.last_motion['global_t_unit'] * s
            self.last_motion['local_t_anchor_metric'] = self.last_motion['local_t_anchor_unit'] * s
            self.last_motion['global_t_anchor_metric'] = self.last_motion['global_t_anchor_unit'] * s
        else:
            self.last_motion['local_t_metric'] = np.full(3, np.nan, dtype = np.float64)
            self.last_motion['global_t_metric'] = np.full(3, np.nan, dtype = np.float64)
            self.last_motion['local_t_anchor_metric'] = np.full(3, np.nan, dtype = np.float64)
            self.last_motion['global_t_anchor_metric'] = np.full(3, np.nan, dtype = np.float64)

    # Обновление метрического масштаба
    def update_metric_scale(self, scale, source = 'stereo_overlap', inliers = 0, residual = np.nan, force = False):
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            return False

        # Защита от резких скачков масштаба
        if self.has_metric_scale() and not force:
            jump = max(scale / self.metric_scale, self.metric_scale / scale)
            if jump > self.metric_scale_max_jump:
                return False

        self.metric_scale = scale
        self.metric_scale_source = source
        self.metric_scale_inliers = int(inliers)
        self.metric_scale_residual = float(residual) if np.isfinite(residual) else np.nan
        self._refresh_metric_fields_in_last_motion()
        
        return True

    # Обработка кадрра
    def process_frame(self, frame):
        self.current_features = self.orb.extract(frame)
        self.last_video_frame = frame.copy()
        self.last_motion_points_for_logging = None
        
        # Инициализация
        if self.state == 'INIT':
            # Сохранение первого кадра
            if self.first_frame is None:
                self.first_frame = frame.copy()
                self.first_features = self.current_features.copy()
                self.pending_init_ready = False

                video_canvas = self._make_video_canvas(frame)
                self._draw_init_result(video_canvas, None, False,
                                       self.cam_id, self.placement, self.frame_idx,
                                       show_empty_message = False)

                self.last_video_canvas = video_canvas
                return video_canvas, None, "init_first_frame"

            # Оценка пары кадров
            result = evaluate_pair(self.camera, self.orb,
                                   self.first_frame, frame,
                                   homo_thresh = self.homo_ransac_threshold,
                                   ess_thresh = self.ess_ransac_threshold,
                                   ess_prob = self.ess_prob,
                                   min_matches = self.min_matches_init,
                                   features1 = self.first_features,
                                   features2 = self.current_features)
            ready = self._is_init_result_ready(result)

            # Результат инициализации
            self.pending_init_result = result if ready else None
            self.pending_init_frame = frame.copy() if ready else None
            self.pending_init_features = self.current_features.copy() if ready else None
            self.pending_init_ready = ready

            if result is not None:
                self.last_init_metrics = {
                    'score': float(result['score']),
                    'parallax_deg': float(result['parallax_deg']),
                    'inlier_count': int(result.get('inlier_count', 0)),
                    'ready': bool(ready)
                }
            else:
                self.last_init_metrics = {'ready': False}

            # Визуализация
            video_canvas = self._make_video_canvas(frame)
            self._draw_init_result(video_canvas, result, ready,
                                   self.cam_id, self.placement, self.frame_idx)
            self.last_video_canvas = video_canvas

            return video_canvas, None, "init_ready" if ready else "init_continue"

        # Отслеживание перемещений
        elif self.state == 'TRACKING':
            res = render_match_and_estimate_pose(
                self.camera, frame,
                self.new_pts3d, self.new_prev_descriptors,
                self.new_rel_R, self.new_rel_t,
                self.new_Glob_R, self.new_Glob_t,
                self.orb,
                search_radius = self.search_radius,
                hamming_threshold = self.hamming_threshold,
                visualize = self.show_tracking_points,
                global_rot_display_deg = self.global_rot_display_deg,
                feature_points = self.current_features.points,
                feature_descriptors = self.current_features.descriptors,
                panel_width = self.video_panel_width,
                point_follow_histories = self.point_follow_histories,
                point_follow_filter_enabled = self.point_follow_filter_enabled,
                point_follow_prev_frames = self.point_follow_prev_frames,
                point_follow_max_prediction_error_px = self.point_follow_max_prediction_error_px,
                point_follow_prediction_error_factor = self.point_follow_prediction_error_factor,
                point_follow_max_step_px = self.point_follow_max_step_px,
                point_follow_step_factor = self.point_follow_step_factor,
                point_follow_min_motion_px = self.point_follow_min_motion_px,
                return_frame = True
            )

            # Проверка успешности отслеивания перемещений
            if res is None or res[7] is False:
                print(f"Cam {self.cam_id}: tracking failed at frame {self.frame_idx}")
                return self.last_video_canvas, self.last_kf_canvas, "track_fail"

            (glob_R, glob_t, rel_R, rel_t, pts3d, desc, good_corrs,
             success, video_canvas, global_rot_display_new, follow_histories_new,
             follow_filter_stats, video_frame) = res

            self.point_follow_histories = follow_histories_new or []

            if video_frame is not None:
                self.last_video_frame = video_frame.copy()

            # Сохранение точек и дескрипторов
            tracked_pixels = np.array([p2d for _, p2d in good_corrs], dtype = np.float64) if good_corrs else np.empty((0, 2), dtype = np.float64)
            self.last_curr_pixels = tracked_pixels
            self.last_curr_points3d_unit = np.asarray(pts3d, dtype = np.float64).reshape(-1, 3) if pts3d is not None else np.empty((0, 3), dtype = np.float64)
            self.last_curr_descriptors = np.asarray(desc, dtype = np.uint8).reshape(-1, 32) if desc is not None else np.empty((0, 32), dtype = np.uint8)

            object_pts_for_logging = (
                np.array([p3d for p3d, _ in good_corrs], dtype = np.float64).reshape(-1, 3)
                if good_corrs else np.empty((0, 3), dtype = np.float64)
            )
            self.last_motion_points_for_logging = {
                'cam_id': self.cam_id,
                'placement': self.placement,
                'frame_idx': self.frame_idx,
                'keyframe_id': self.current_kf.id if self.current_kf is not None else -1,
                'pixels': tracked_pixels.astype(np.float64, copy = True),
                'pts3d_unit_input': object_pts_for_logging,
                'pts3d_unit_refined': self.last_curr_points3d_unit.astype(np.float64, copy = True),
                'descriptors': self.last_curr_descriptors.astype(np.uint8, copy = True),
                'source': 'tracking_pose_estimation',
            }

            if not success:
                return self.last_video_canvas, self.last_kf_canvas, "track_fail"

            # Обновление глобальных параметров
            self.current_kf.dt = rel_R @ self.current_kf.dt + rel_t
            self.current_kf.dR = rel_R @ self.current_kf.dR
            self.new_Glob_R = glob_R
            self.new_Glob_t = glob_t
            self.new_rel_R = rel_R
            self.new_rel_t = rel_t
            self.global_rot_display_deg = normalize_angles_360(global_rot_display_new)

            # расстояние между двумя ключевыми кадрами
            baseline_unit = float(np.linalg.norm(np.asarray(self.current_kf.dt, dtype = np.float64).reshape(3)))
            baseline_rel = np.nan
            old_points_for_baseline = np.asarray(getattr(self.current_kf, 'pts3d', np.empty((0, 3))), dtype = np.float64).reshape(-1, 3)
            if len(old_points_for_baseline):
                old_norm = np.linalg.norm(old_points_for_baseline, axis = 1)
                old_norm = old_norm[np.isfinite(old_norm) & (old_norm > 1e-9)]
                if len(old_norm):
                    baseline_rel = float(baseline_unit / (np.median(old_norm) + 1e-12))

            # Обновление данных о перемещении
            num_inliers = len(good_corrs)
            self._update_last_motion(rel_R, rel_t, glob_R, glob_t, num_inliers,
                                     global_rot_display = self.global_rot_display_deg)

            # Обновление кадра
            if video_canvas is not None:
                self.last_video_canvas = video_canvas
                refreshed = self.refresh_last_video_canvas_panel()
                if refreshed is not None:
                    video_canvas = refreshed

            # Создание нового ключевого кадра
            accepted_new_kf = False
            if num_inliers < self.min_matches_kf:
                new_kf = create_new_keyframe(
                    self.current_kf, frame,
                    self.current_kf.dR, self.current_kf.dt,
                    self.camera, self.orb,
                    kf_id = self.kf_counter + 1,
                    min_matches = self.kf_min_matches,
                    max_reproj_error = self.kf_max_reproj_error,
                    curr_features = self.current_features
                )
                if new_kf is not None:
                    accepted_new_kf = True
                    self.kf_counter += 1
                    self.current_kf = new_kf
                    self.new_pts3d = new_kf.pts3d
                    self.new_prev_descriptors = new_kf.descriptors
                    self.last_curr_pixels = new_kf.pts2d.astype(np.float64, copy = True)
                    self.last_curr_points3d_unit = new_kf.pts3d.astype(np.float64, copy = True)
                    self.last_curr_descriptors = new_kf.descriptors.astype(np.uint8, copy = True)
                    self.point_follow_histories = make_point_follow_histories(
                        self.last_curr_pixels, self.point_follow_prev_frames
                    )
                    
                    # Сохранение ключевого кадра
                    self.last_keyframe_event = {
                        'cam_id': self.cam_id,
                        'placement': self.placement,
                        'frame_idx': self.frame_idx,
                        'keyframe_id': new_kf.id,
                        'pts2d': new_kf.pts2d.astype(np.float64, copy = True),
                        'pts3d_unit': new_kf.pts3d.astype(np.float64, copy = True),
                        'descriptors': new_kf.descriptors.astype(np.uint8, copy = True),
                        'source': 'tracking_keyframe',
                    }
                    self.new_rel_R = rel_R.copy()
                    self.new_rel_t = rel_t.copy()
                    self.new_Glob_R = new_kf.R
                    self.new_Glob_t = new_kf.t

                    # Визуализация ключевого кадра
                    self.last_kf_canvas = render_keyframe_reprojection(
                        self.camera, new_kf.frame, new_kf.pts2d, new_kf.pts3d,
                        new_kf.id, additional_info = f"place: {self.placement}",
                        panel_width = self.kf_panel_width,
                        font_scale = self.display_font_scale,
                        thickness = self.display_font_thickness,
                        line_height = self.display_line_height,
                        margin_x = self.display_margin_x,
                        start_y = self.display_y_start
                    )

            # Если нет ключевого кадра
            if not accepted_new_kf:
                self.new_pts3d = np.asarray(pts3d, dtype = np.float64).reshape(-1, 3) if pts3d is not None else np.empty((0, 3), dtype = np.float64)
                self.new_prev_descriptors = np.asarray(desc, dtype = np.uint8).reshape(-1, 32) if desc is not None else np.empty((0, 32), dtype = np.uint8)

            if self.last_motion is not None:
                keyframe_id_for_log = self.current_kf.id if self.current_kf is not None else -1
                self.last_motion.update({
                    'is_keyframe': bool(accepted_new_kf),
                    'keyframe_id': int(keyframe_id_for_log),
                    'baseline_unit': float(baseline_unit),
                    'baseline_rel': float(baseline_rel) if np.isfinite(baseline_rel) else np.nan,
                    'min_baseline_unit': float(KEYFRAME_CONFIG.get('min_baseline_unit', np.nan)),
                    'min_baseline_rel': float(KEYFRAME_CONFIG.get('min_baseline_rel', np.nan)),
                    'motion_points_count': int(follow_filter_stats.get('matched_before_follow', num_inliers)),
                    'filtered_points_count': int(follow_filter_stats.get('matched_after_follow', num_inliers)),
                    'rotation_points_count': int(num_inliers),
                    'translation_points_count': int(num_inliers),
                    'follow_filter_enabled': bool(follow_filter_stats.get('enabled', False)),
                    'follow_filter_prev_frames': int(follow_filter_stats.get('prev_frames', self.point_follow_prev_frames)),
                    'follow_filter_rejected': int(follow_filter_stats.get('rejected_by_follow', 0)),
                })

            self.last_video_canvas = video_canvas
            return video_canvas, self.last_kf_canvas, "tracking"

        return None, None, "unknown"

    # Закрытие видео
    def release(self):
        self.cap.release()

# Оценка позиции кадра
def render_match_and_estimate_pose(camera, frame3,
                                   pts3d_in_cam2, desc, R12, t12, global_R, global_t, orb,
                                   search_radius = POSE_MATCH_CONFIG['search_radius'],
                                   hamming_threshold = POSE_MATCH_CONFIG['hamming_threshold'],
                                   visualize = POSE_MATCH_CONFIG['visualize'],
                                   global_rot_display_deg = None,
                                   feature_points = None, feature_descriptors = None,
                                   panel_width = POSE_MATCH_CONFIG['panel_width'],
                                   point_follow_histories = None,
                                   point_follow_filter_enabled = False,
                                   point_follow_prev_frames = 3,
                                   point_follow_max_prediction_error_px = 25.0,
                                   point_follow_prediction_error_factor = 4.0,
                                   point_follow_max_step_px = 80.0,
                                   point_follow_step_factor = 4.0,
                                   point_follow_min_motion_px = 1.0,
                                   return_frame = False):
    # 3d-точки в пиксели
    rays = camera.point_to_ray_vectorized(pts3d_in_cam2, R = R12, t = t12.reshape(3))
    projected_pts = camera.ray_to_pixel_vectorized(rays)

    # Поиск точек на текущем кадре
    if feature_points is None or feature_descriptors is None:
        features3 = orb.extract(frame3)
        points3, desc3 = features3.points, features3.descriptors
    else:
        points3 = np.asarray(feature_points, dtype = np.float32).reshape(-1, 2)
        desc3 = np.asarray(feature_descriptors, dtype = np.uint8)
    if len(points3) == 0 or desc3 is None or len(desc3) == 0:
        print("На кадре 3 нет точек")
        return None, None, None, None, None, None, None, False, None, None, [], {}, None
    keypoints_xy = points3.astype(np.float32, copy = False)
    h, w = frame3.shape[:2]

    # Деление кадра на сетку для быстрого поиска
    cell_size = max(4.0, search_radius * 0.5)
    grid_cols = int(np.ceil(w / cell_size))
    grid_rows = int(np.ceil(h / cell_size))
    grid = [[[] for _ in range(grid_cols)] for _ in range(grid_rows)]
    for idx, (x, y) in enumerate(keypoints_xy):
        cx = int(x // cell_size)
        cy = int(y // cell_size)
        if 0 <= cx < grid_cols and 0 <= cy < grid_rows:
            grid[cy][cx].append(idx)
    used = set()
    correspondences = []
    follow_filter_stats = {
        'enabled': bool(point_follow_filter_enabled),
        'prev_frames': int(point_follow_prev_frames),
        'matched_before_follow': 0,
        'matched_after_follow': 0,
        'rejected_by_follow': 0,
    }
    n_histories = len(point_follow_histories) if point_follow_histories is not None else 0

    # Поиск соответствий между репроецированными точками и найденными на текущем кадре
    for i, proj in enumerate(projected_pts):
        if np.isnan(proj[0]) or np.isnan(proj[1]):
            continue
        cx = int(proj[0] // cell_size)
        cy = int(proj[1] // cell_size)
        candidate_indices = set()
        cell_span = int(np.ceil(search_radius / cell_size))
        for dy in range(-cell_span, cell_span + 1):
            for dx in range(-cell_span, cell_span + 1):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < grid_rows and 0 <= nx < grid_cols:
                    candidate_indices.update(grid[ny][nx])
        candidate_indices -= used # Исключение найденных соответствий
        candidates = []
        for idx in candidate_indices:
            x, y = keypoints_xy[idx]
            dist_geo = np.hypot(proj[0] - x, proj[1] - y)
            if dist_geo <= search_radius:
                hamming_dist = cv2.norm(desc[i], desc3[idx], cv2.NORM_HAMMING)
                if hamming_dist <= hamming_threshold:
                    candidates.append((idx, hamming_dist))

        if not candidates:
            continue

        # Проверка естественности движения точек
        follow_filter_stats['matched_before_follow'] += 1
        candidates.sort(key = lambda x: x[1])
        accepted_idx = None
        for cand_idx, cand_hamming in candidates:
            current_pixel = keypoints_xy[cand_idx]
            if point_follow_filter_enabled and i < n_histories:
                ok_follow = point_follow_motion_is_natural(
                    point_follow_histories[i],
                    current_pixel,
                    point_follow_prev_frames,
                    point_follow_max_prediction_error_px,
                    point_follow_prediction_error_factor,
                    point_follow_max_step_px,
                    point_follow_step_factor,
                    point_follow_min_motion_px,
                )
                if not ok_follow:
                    continue

            accepted_idx = cand_idx
            break

        if accepted_idx is None:
            follow_filter_stats['rejected_by_follow'] += 1
            continue

        used.add(accepted_idx)
        correspondences.append((pts3d_in_cam2[i], keypoints_xy[accepted_idx], accepted_idx, i))

    follow_filter_stats['matched_after_follow'] = len(correspondences)
    
    if point_follow_filter_enabled:
        print(
            f"\nНайдено соответствий: {len(correspondences)} "
            f"(до follow-фильтра: {follow_filter_stats['matched_before_follow']}, "
            f"отклонено: {follow_filter_stats['rejected_by_follow']})"
        )
    else:
        print(f"\nНайдено соответствий: {len(correspondences)}")
        
    min_correspondences = int(PNP_CONFIG.get('min_correspondences', 4))
    if len(correspondences) < min_correspondences:
        print(f"Недостаточно соответствий для оценки позы (нужно минимум {min_correspondences})")
        return None, None, None, None, None, None, None, False, None, None, [], {}, None

    # Данные для PnP
    object_pts = np.array([c[0] for c in correspondences], dtype = np.float64)
    image_pts = np.array([c[1] for c in correspondences], dtype = np.float64)
    rays = camera.pixel_to_ray_vectorized(image_pts)
    normalized_pts = camera.ray_to_plane_vectorized(rays)
    valid_uv = ~np.isnan(normalized_pts).any(axis = 1)
    object_pts_valid = object_pts[valid_uv]
    image_pts_norm = normalized_pts[valid_uv]
    valid_indices = np.where(valid_uv)[0]
    if len(object_pts_valid) < min_correspondences:
        print("После нормализации недостаточно точек")
        return None, None, None, None, None, None, None, False, None, None, [], {}, None

    # Решение задачи PnP RANSAC
    K = np.eye(3, dtype = np.float64)
    dist_coeffs = np.zeros(4)
    pnp_thresh_norm = float(PNP_CONFIG.get('ransac_reprojection_error_norm', 0.02))
    success, rvec, tvec, inliers_ransac = cv2.solvePnPRansac(
        object_pts_valid, image_pts_norm, K, dist_coeffs,
        iterationsCount = int(PNP_CONFIG.get('ransac_iterations_count', 200)),
        reprojectionError = pnp_thresh_norm,
        confidence = float(PNP_CONFIG.get('ransac_confidence', 0.99)),
        flags = cv2.SOLVEPNP_EPNP
    )
    if not success or inliers_ransac is None or len(inliers_ransac) < min_correspondences:
        print("RANSAC не дал решения")
        return None, None, None, None, None, None, None, False, None, None, [], {}, None

    # Уточнение позиции с помощью PnP Левенберг-Марквардт
    obj_inliers_ransac = object_pts_valid[inliers_ransac.flatten()]
    img_inliers_ransac = image_pts_norm[inliers_ransac.flatten()]
    rvec_opt, tvec_opt = rvec.copy(), tvec.copy()
    rvec_opt, tvec_opt = cv2.solvePnPRefineLM(
        obj_inliers_ransac, img_inliers_ransac, K, dist_coeffs,
        rvec_opt, tvec_opt,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
         int(PNP_CONFIG.get('refine_lm_max_iter', 30)),
         float(PNP_CONFIG.get('refine_lm_eps', 1e-7)))
    )
    K_eye = np.eye(3)
    dist_none = np.zeros(4)
    for _ in range(int(PNP_CONFIG.get('ba_iterations', 3))):
        R_opt, _ = cv2.Rodrigues(rvec_opt)
        t_opt = tvec_opt.reshape(3)
        rays = camera.point_to_ray_vectorized(object_pts_valid, R = R_opt, t = t_opt)
        proj_all = camera.ray_to_plane_vectorized(rays)
        errors_all = np.linalg.norm(proj_all - image_pts_norm, axis = 1)
        ba_inliers = errors_all < pnp_thresh_norm
        if np.sum(ba_inliers) < min_correspondences:
            break
        rvec_new, tvec_new = cv2.solvePnPRefineLM(
            object_pts_valid[ba_inliers], image_pts_norm[ba_inliers],
            K_eye, dist_none, rvec_opt, tvec_opt,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
             int(PNP_CONFIG.get('ba_refine_max_iter', 20)),
             float(PNP_CONFIG.get('ba_refine_eps', 1e-6)))
        )
        if np.linalg.norm(rvec_new - rvec_opt) < 1e-6 and np.linalg.norm(tvec_new - tvec_opt) < 1e-6:
            break
        rvec_opt, tvec_opt = rvec_new, tvec_new

    # Глобальная позиция кадра
    R23, _ = cv2.Rodrigues(rvec_opt)
    t23 = tvec_opt.reshape(3)
    global_t_new = R23 @ global_t + t23
    global_R_new = R23 @ global_R
    rel_rot_deg = np.array(rotation_matrix_to_euler(R23, degrees = True), dtype = np.float64)
    rel_rot_delta = signed_delta_angles_deg(rel_rot_deg)
    if global_rot_display_deg is None:
        global_rot_display_new = normalize_angles_360(rel_rot_delta)
    else:
        global_rot_display_new = normalize_angles_360(
            np.asarray(global_rot_display_deg, dtype = np.float64) + rel_rot_delta
        )
    R23, _ = cv2.Rodrigues(rvec_opt)
    t23 = tvec_opt.reshape(3)

    # Фильтрация по ошибке репроекции
    rays_final = camera.point_to_ray_vectorized(object_pts_valid, R = R23, t = t23)
    proj_norm_final = camera.ray_to_plane_vectorized(rays_final)
    final_norm_errors = np.linalg.norm(proj_norm_final - image_pts_norm, axis = 1)
    proj_pix_final = camera.ray_to_pixel_vectorized(rays_final)
    image_pts_pix_valid = image_pts[valid_indices]
    final_pix_errors = np.linalg.norm(proj_pix_final - image_pts_pix_valid, axis = 1)
    final_pixel_threshold = max(
        float(PNP_CONFIG.get('final_pixel_error_min', 4.0)),
        search_radius * float(PNP_CONFIG.get('final_pixel_error_radius_factor', 0.5))
    )
    final_inliers_mask = (final_norm_errors < pnp_thresh_norm) & (final_pix_errors < final_pixel_threshold)
    if np.sum(final_inliers_mask) < min_correspondences:
        print("После BA недостаточно точек")
        return None, None, None, None, None, None, None, False, None, None, [], {}, None

    # Результаты
    inlier_valid_indices = np.where(final_inliers_mask)[0]
    orig_indices = valid_indices[inlier_valid_indices]
    p3d_cam2_inliers = object_pts[orig_indices]
    p2d_inliers = image_pts[orig_indices]
    idx_kp_inliers = np.fromiter(
        (correspondences[idx][2] for idx in orig_indices),
        dtype = np.int32,
        count = len(orig_indices)
    )
    src_point_indices = np.fromiter(
        (correspondences[idx][3] for idx in orig_indices),
        dtype = np.int32,
        count = len(orig_indices)
    )
    points3d_cam3 = (R23 @ p3d_cam2_inliers.T).T + t23.reshape(1, 3)
    descriptors_cam3 = desc3[idx_kp_inliers]
    good_corrs = list(zip(p3d_cam2_inliers, p2d_inliers))

    # Обновление истории движения точек
    next_point_follow_histories = []
    for src_idx, pixel in zip(src_point_indices, p2d_inliers):
        prev_history = point_follow_histories[src_idx] if point_follow_histories is not None and src_idx < len(point_follow_histories) else []
        next_point_follow_histories.append(
            extend_point_follow_history(prev_history, pixel, point_follow_prev_frames)
        )

    # Визуализация результатов
    display_frame = frame3.copy()
    if visualize:
        for x, y in keypoints_xy:
            u, v = int(x), int(y)
            if 0 <= u < display_frame.shape[1] and 0 <= v < display_frame.shape[0]:
                cv2.circle(display_frame, (u, v), 3, (0, 255, 0), -1)
        if good_corrs:
            p3d_array = np.array([p for p, _ in good_corrs], dtype = np.float64)
            rays = camera.point_to_ray_vectorized(p3d_array, R = R23, t = t23)
            proj_pix_array = camera.ray_to_pixel_vectorized(rays)
            for idx, (_, p2d) in enumerate(good_corrs):
                proj_pix = proj_pix_array[idx]
                if not np.isnan(proj_pix[0]):
                    u_proj, v_proj = int(round(proj_pix[0])), int(round(proj_pix[1]))
                    u_orb, v_orb = int(p2d[0]), int(p2d[1])
                    in_proj = 0 <= u_proj < display_frame.shape[1] and 0 <= v_proj < display_frame.shape[0]
                    in_orb = 0 <= u_orb < display_frame.shape[1] and 0 <= v_orb < display_frame.shape[0]
                    if in_proj:
                        cv2.circle(display_frame, (u_proj, v_proj), 3, (255, 0, 0), -1)
                    if in_proj and in_orb:
                        cv2.line(display_frame, (u_proj, v_proj), (u_orb, v_orb), (0, 0, 255), 1)
    h, w = frame3.shape[:2]
    panel_width = max(0, int(panel_width))
    canvas = np.ones((h, w + panel_width, 3), dtype = np.uint8) * 255
    canvas[:, panel_width:] = display_frame

    result = (global_R_new, global_t_new, R23, t23, points3d_cam3, descriptors_cam3,
              good_corrs, True, canvas, global_rot_display_new,
              next_point_follow_histories, follow_filter_stats)
    if return_frame:
        return result + (display_frame,)
    return result


# Ключевые кадры
class KeyFrame:
    def __init__(self, kf_id, frame, pts2d, pts3d, descriptors, R, t, dR, dt,
                 feature_points = None, feature_descriptors = None):
        self.id = kf_id
        self.frame = frame.copy()
        self.pts2d = pts2d.copy()
        self.pts3d = pts3d.copy()
        self.descriptors = descriptors.copy()
        self.R = R.copy()
        self.t = t.copy()
        self.dR = dR.copy()
        self.dt = dt.copy()
        if feature_points is None or feature_descriptors is None:
            self.feature_points = self.pts2d.astype(np.float32, copy = True)
            self.feature_descriptors = self.descriptors.astype(np.uint8, copy = True)
        else:
            self.feature_points = np.asarray(feature_points, dtype = np.float32).reshape(-1, 2).copy()
            self.feature_descriptors = np.asarray(feature_descriptors, dtype = np.uint8).copy()

    def features(self) -> FeatureSet:
        return FeatureSet(self.feature_points, self.feature_descriptors)


# Создание ключевых кадров
def create_new_keyframe(old_kf, curr_frame, R_rel_accum, t_rel_accum,
                        camera, orb, kf_id, min_matches = KEYFRAME_CONFIG['min_matches'],
                        max_reproj_error = KEYFRAME_CONFIG['max_reproj_error'],
                        curr_features: Optional[FeatureSet] = None):
    # Расстояние между ключевыми кадрами
    baseline = float(np.linalg.norm(np.asarray(t_rel_accum, dtype = np.float64).reshape(3)))
    if not np.isfinite(baseline):
        return None
    min_baseline = float(KEYFRAME_CONFIG.get('min_baseline_unit', 0.0))
    if min_baseline > 0.0 and baseline < min_baseline:
        return None

    old_points = np.asarray(getattr(old_kf, 'pts3d', np.empty((0, 3))), dtype = np.float64).reshape(-1, 3)
    old_norm = np.linalg.norm(old_points, axis = 1) if len(old_points) else np.empty(0, dtype = np.float64)
    old_norm = old_norm[np.isfinite(old_norm) & (old_norm > 1e-9)]

    if len(old_norm):
        old_med_norm = float(np.median(old_norm))
        baseline_rel = baseline / (old_med_norm + 1e-12)
        min_baseline_rel = float(KEYFRAME_CONFIG.get('min_baseline_rel', 0.0))
        if min_baseline_rel > 0.0 and baseline_rel < min_baseline_rel:
            return None

    # Соответствия между ключевыми кадрами
    if curr_features is None:
        curr_features = orb.extract(curr_frame)
    old_features = old_kf.features() if hasattr(old_kf, 'features') else orb.extract(old_kf.frame)
    pts1_pix, pts2_pix, matches = orb.match_feature_sets(old_features, curr_features)
    if len(pts1_pix) < min_matches:
        print(f"Недостаточно совпадений: {len(pts1_pix)}")
        return None
    train_idx = np.fromiter((m.trainIdx for m in matches), dtype = np.int32, count = len(matches))
    desc_curr_matched = curr_features.descriptors[train_idx]

    # Нормализация координат
    rays1 = camera.pixel_to_ray_vectorized(pts1_pix)
    uv1 = camera.ray_to_plane_vectorized(rays1)
    rays2 = camera.pixel_to_ray_vectorized(pts2_pix)
    uv2 = camera.ray_to_plane_vectorized(rays2)
    
    pts3d_old_all = triangulate_points(uv1, uv2, R_rel_accum, t_rel_accum, K = np.eye(3))
    pts3d_new_all = (R_rel_accum @ pts3d_old_all.T).T + t_rel_accum.reshape(1, 3)

    # Фильтрация по глубине
    valid_depth = (pts3d_old_all[:, 2] > 0) & (pts3d_new_all[:, 2] > 0)
    if np.sum(valid_depth) < KEYFRAME_CONFIG.get('min_positive_depth_points', 10):
        print("Слишком мало точек с положительной глубиной")
        return None
    
    pts3d_old = pts3d_old_all[valid_depth]
    pts3d_new = pts3d_new_all[valid_depth]
    pts1_pix_v = pts1_pix[valid_depth]
    pts2d_curr = pts2_pix[valid_depth]
    desc_curr = desc_curr_matched[valid_depth]

    # Ошибки репроекций
    reproj_old = camera.ray_to_pixel_vectorized(camera.point_to_ray_vectorized(pts3d_old))
    reproj_new = camera.ray_to_pixel_vectorized(camera.point_to_ray_vectorized(pts3d_new))

    # Проверка границ изображения
    h, w = curr_frame.shape[:2]
    in_bounds_new = (reproj_new[:, 0] >= 0) & (reproj_new[:, 0] < w - 1) & \
                    (reproj_new[:, 1] >= 0) & (reproj_new[:, 1] < h - 1)
    err_old = np.linalg.norm(reproj_old - pts1_pix_v, axis = 1)
    err_new = np.linalg.norm(reproj_new - pts2d_curr, axis = 1)

    # Вычисление параллакса
    rays2_in_old = (R_rel_accum.T @ rays2[valid_depth].T).T
    rays1_v = rays1[valid_depth]
    rays1_v /= np.linalg.norm(rays1_v, axis = 1, keepdims = True) + 1e-12
    rays2_in_old /= np.linalg.norm(rays2_in_old, axis = 1, keepdims = True) + 1e-12
    cos_par = np.sum(rays1_v * rays2_in_old, axis = 1)
    parallax_deg = np.degrees(np.arccos(np.clip(cos_par, -1.0, 1.0)))

    # Комплексная фильтрация
    keep_mask = (in_bounds_new &
                 (err_old <= max_reproj_error) &
                 (err_new <= max_reproj_error) &
                 (parallax_deg >= KEYFRAME_CONFIG.get('min_parallax_deg', 0.5)))
    n_good = int(np.sum(keep_mask))
    print(f"После фильтрации KF: {n_good} / {len(pts3d_new)} точек")
    if n_good > 0:
        print(f"err_old mean/max: {np.mean(err_old[keep_mask]):.2f}/{np.max(err_old[keep_mask]):.2f} px; "
              f"err_new mean/max: {np.mean(err_new[keep_mask]):.2f}/{np.max(err_new[keep_mask]):.2f} px; "
              f"parallax median: {np.median(parallax_deg[keep_mask]):.2f} deg")
    if n_good < KEYFRAME_CONFIG.get('min_good_points', 10):
        print("Слишком мало хороших точек – ключевой кадр не создан")
        return None

    # Созддание нового ключевого кадра
    pts3d_filtered = pts3d_new[keep_mask]
    pts2d_filtered = pts2d_curr[keep_mask]
    desc_filtered = desc_curr[keep_mask]
    R_glob = R_rel_accum @ old_kf.R
    t_glob = R_rel_accum @ old_kf.t + t_rel_accum
    new_kf = KeyFrame(kf_id,
                      curr_frame,
                      pts2d_filtered,
                      pts3d_filtered,
                      desc_filtered,
                      R_glob,
                      t_glob,
                      np.eye(3),
                      np.zeros(3),
                      feature_points = curr_features.points,
                      feature_descriptors = curr_features.descriptors)
    print(f"Новый ключевой {kf_id} кадр создан: {len(pts3d_filtered)} хороших точек")
    
    return new_kf

# Инициализация
def evaluate_pair(camera, orb, frame1, frame2,
                  homo_thresh = EVALUATE_PAIR_CONFIG['homo_thresh'],
                  ess_thresh = EVALUATE_PAIR_CONFIG['ess_thresh'],
                  ess_prob = EVALUATE_PAIR_CONFIG['ess_prob'],
                  min_matches = EVALUATE_PAIR_CONFIG['min_matches'],
                  features1: Optional[FeatureSet] = None,
                  features2: Optional[FeatureSet] = None):
    if features1 is None:
        features1 = orb.extract(frame1)
    if features2 is None:
        features2 = orb.extract(frame2)
        
    pts1_pix, pts2_pix, matches = orb.match_feature_sets(features1, features2)
    desc1, desc2 = features1.descriptors, features2.descriptors
    
    if len(pts1_pix) < min_matches:
        return None
    
    rays1 = camera.pixel_to_ray_vectorized(pts1_pix)
    uv1 = camera.ray_to_plane_vectorized(rays1)
    rays2 = camera.pixel_to_ray_vectorized(pts2_pix)
    uv2 = camera.ray_to_plane_vectorized(rays2)
    
    homo = Search_homography(ransac_threshold = homo_thresh)
    homo.compute(uv1, uv2)
    ess = Search_essential_matrix(ransac_threshold = ess_thresh, prob = ess_prob)
    ess.compute(uv1, uv2)
    
    best = select_best_model(homo, ess, uv1, uv2)
    if best is None:
        return None
    
    inlier_mask = best['mask']
    inlier_count = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
    pts1_in = pts1_pix[inlier_mask]
    pts2_in = pts2_pix[inlier_mask]
    
    if len(pts1_in) < 4:
        parallax_deg = 0.0
    else:
        rays1_in = camera.pixel_to_ray_vectorized(pts1_in)
        rays2_in = camera.pixel_to_ray_vectorized(pts2_in)
        norms1 = np.linalg.norm(rays1_in, axis = 1, keepdims = True)
        norms2 = np.linalg.norm(rays2_in, axis = 1, keepdims = True)
        rays1_n = rays1_in / norms1
        rays2_n = rays2_in / norms2
        dots = np.sum(rays1_n * rays2_n, axis = 1)
        dots = np.clip(dots, -1.0, 1.0)
        angles_rad = np.arccos(dots)
        parallax_deg = np.median(angles_rad) * 180.0 / np.pi
        
    return {
        'success': True,
        'score': best['score'],
        'model_type': 'Homography' if best['rh'] > 0.5 else 'Essential',
        'rh': best['rh'],
        'parallax_deg': parallax_deg,
        'inlier_count': inlier_count,
        'R': best['R'],
        't': best['t'],
        'pts1_uv': best['pts1_uv'],
        'pts2_uv': best['pts2_uv'],
        'mask': best['mask'],
        'pts1_pix': pts1_pix,
        'pts2_pix': pts2_pix,
        'matches': matches,
        'desc1': desc1,
        'desc2': desc2
    }
