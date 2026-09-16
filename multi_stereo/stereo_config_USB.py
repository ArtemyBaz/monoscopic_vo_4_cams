import numpy as np

# ------------------- ОБЩИЕ НАСТРОЙКИ -------------------
APP_CONFIG = {
    'max_cams': 4, # Количество камер
    'default_frame_size': (480, 640), # Размер кадра
    'anchor_camera_id': 1, # Опорная камера
}

# ------------------- НАСТРОЙКИ ORB -------------------
ORB_DEFAULTS = {
    'nfeatures': 2000, # Максимальное количество точек при поиске
    'scaleFactor': 1.2, # Масштаб уровней пирамиды
    'nlevels': 8, # Количество уровней пирамиды
    'edgeThreshold': 31, # расстояния от края, где поиск точек не происходит
    'firstLevel': 0, # Первый уровень пирамиды
    'WTA_K': 2, # Колическтво бит для дескриптора
    'scoreType': 'HARRIS', 
    'patchSize': 31, # Область вокруг точки для вычисления  дескриптора
    'fastThreshold': 20, # Порог детектора
}

# ------------------- НАСТРОЙКИ ОТСЛЕЖИВАНИЯ ПЕРЕМЕЩЕНИЙ -------------------
TRACKER_DEFAULTS = {
    'placement': 'front', # Размещение камеры: 'front', 'back', 'left', 'right'
    'nfeatures': ORB_DEFAULTS['nfeatures'],
    'scaleFactor': ORB_DEFAULTS['scaleFactor'],
    'nlevels': ORB_DEFAULTS['nlevels'],
    'score_threshold': 5000.0, # Минимальный score для инициализации
    'min_parallax_deg': 4.0, # Минимальный параллакс
    'min_matches_init': 400, # Минимальное количество совпадений для инициализации
    'min_inliers_init': 400, # Минимальное количество хороших точек для инициализации
    'normalize_init_scale': True, # Приведение масштаба к единице
    'homo_ransac_threshold': 0.05, # Порог RANSAC для гомографии
    'ess_ransac_threshold': 0.008, # Порог RANSAC для существенной матрице
    'ess_prob': 0.999, # Доверительная вероятность для RANSAC
    'search_radius': 10, # Радиус поиска совпадений между репроецированными и найденными точками
    'hamming_threshold': 300, # Максимальное расстояние Хэмминга для соответствия точек
    'min_matches_kf': 200, # Количество точек при котором необходимо создать новый ключевой кадр
    'kf_min_matches': 100, # Минимальное количество совпадений для создания нового ключевого кадра 
    'kf_max_reproj_error': 5.0, # Максимальная пиксельная ошибка репроекции при создании ключевого кадра
    'metric_scale_alpha': 0.15, # Коэффициент сглаживания метрического масштаба
    'metric_scale_max_jump': 1.2, # Максимальное изменение между коэффициентами масштаба, больше которого масштаб не применяется
    'point_follow_filter_enabled': True, # Включение проверки естественности движения точек
    'point_follow_prev_frames': 3, # Количество предыдущих кадров для истории точек
    'point_follow_max_prediction_error_px': 5.0, # Максимальная ошибка предсказания в пикселях
    'point_follow_prediction_error_factor': 4.0, # Множитель для динамического порога предсказания
    'point_follow_max_step_px': 2.0, # Максимальный пиксельный шаг точким
    'point_follow_step_factor': 4.0, # Множитель для динамического порога шага
    'point_follow_min_motion_px': 1.0, # Минимальное движение для анализа
}

TRACKER_DISPLAY_DEFAULTS = {
    'panel_width': 220, # Ширина информационной панели справа от видео в пикселя
    'font_scale': 0.5, # Размер шрифта на панели
    'font_thickness': 1, # Толщина шрифта
    'display_margin_x': 10, # Отступ текста от левого края панели
    'display_y_start': 24, # Начальная позиция текста по Y
    'display_line_height': 20, # Высота строки текста
    'display_section_gap': 8, # Отступ между секциями информации
}

# ------------------- НАСТРОЙКИ ГОМОГРАФИИ И СУЩЕСТВЕННОЙ МАТРИЦЫ ДЛЯ ИНИЦИАЛИЗАЦИИ -------------------
MODEL_ESTIMATION_DEFAULTS = {
    'homography_ransac_threshold': 1.0, # Порог RANSAC для гомографии
    'essential_ransac_threshold': 1.0,  # Порог RANSAC для существенной матрицы
    'essential_prob': 0.999, # Доверительная вероятность для RANSAC существенной матрицы
}

EVALUATE_PAIR_CONFIG = {
    'homo_thresh': 0.05, # Порог для гомографии при оценке пары кадров
    'ess_thresh': 0.008,  # Порог для существенной матрицы при оценке пары кадров
    'ess_prob': 0.999, # Доверительная вероятность для существенной матрицы
    'min_matches': 500, # Минимальное количество совпадений для оценки пары кадров
}

# ------------------- НАСТРОЙКИ PNP -------------------
POSE_MATCH_CONFIG = {
    'search_radius': 10, # Радиус поиска соответствий в пикселях
    'hamming_threshold': 200, # Максимальное расстояние Хэмминга для сопоставления
    'visualize': True, # Визуализация сопоставлений
    'panel_width': 220, # Ширина панели при визуализации
}

PNP_CONFIG = {
    'min_correspondences': 4, # Минимальное количество соответствий для PnP
    'ransac_iterations_count': 200, # Количество итераций RANSAC
    'ransac_reprojection_error_norm': 0.02, # Порог ошибки репроекции в нормализованных координатах
    'ransac_confidence': 0.99, # Доверительная вероятность RANSAC
    'refine_lm_max_iter': 30, # Максимальное количество итераций PnP Левенберга-Марквардта
    'refine_lm_eps': 1e-7,  # Критерий сходимости для LM
    'ba_iterations': 12,  # Количество итераций LM
    'ba_refine_max_iter': 20, # Максимальное количество итераций при уточнении LM
    'ba_refine_eps': 1e-6, # Критерий сходимости
    'final_pixel_error_min': 4.0, # Минимальный порог ошибки репроекции в пикселях
    'final_pixel_error_radius_factor': 0.5, # Множитель радиуса поиска для динамического порога
}

# ------------------- НАСТРОЙКИ КЛЮЧЕВЫХ КАДРОВ -------------------
KEYFRAME_CONFIG = {
    'min_matches': 100, # Минимальное количество совпадений для создания ключевого кадра
    'max_reproj_error': 5.0, # Максимальная ошибка репроекции в пикселях
    'min_positive_depth_points': 10, # Минимальное количество точек с положительной глубиной
    'min_good_points': 10, # Минимальное количество хороших точек после фильтрации
    'min_parallax_deg': 0.5, # Минимальный параллакс для ключевого кадра
    'min_baseline_unit': 0.04, # Минимальное расстояние в условных координатах
    'min_baseline_rel': 0.04, # Минимальное расстояниев реальных координатах
}

# ------------------- НАСТРОЙКИ СТЕРЕО -------------------
CROSS_PAIRS = [
    (1, 3, (255, 0, 0), "F-R"),
    (3, 2, (0, 255, 0), "R-B"),
    (2, 0, (0, 0, 255), "B-L"),
    (0, 1, (255, 255, 0), "L-F"),
]

CROSS_MATCH_CONFIG = {
    'hamming_threshold': 200, # Порог Хэмминга для сопоставления между камерами
}

GEOMETRY_CONFIG = {
    'use_geometric_filter': True, # Включение геометрической фильтрации
    'epipolar_threshold': 7.0, # Порог фильтрации
}

# ------------------- НАСТРОЙКИ ЗАПИСИ ВИДЕО -------------------
RECORDING_CONFIG = {
    'save_final_window': True, # Включение сохранения видео
    'save_keyframe_reprojection': True, # Включение сохранения ключевых кадров с репроекциями
    'final_window_path': 'outputs/final_window.avi', # Путь сохранения общего видео
    'keyframe_reprojection_path': 'outputs/keyframe_reprojection.avi',  # Путь сохранения видео с ключевыми кадрами
    'fps': None, # FPS сохранения
    'fallback_fps': 30.0, # FPS сохранения по умолчанию
    'fourcc': 'XVID', # Кодек для видео
    'keyframe_grid_cols': 2, # Количество колонок в сетке репроекций ключевых кадров
    'resize_on_size_change': True, # Включение изменения размера кадра при изменении размера видео
    'print_status': True, # Вывод статуса записи в консоль
}


# ------------------- НАСТРОЙКИ CSV-ЛОГИРОВАНИЯ И ПАПКИ ЗАПУСКА -------------------
LOGGING_CONFIG = {
    'enabled': True, # Включение логирования

    # Папка для сохранения логов
    'root_dir': 'outputs', 
    'directory_prefix': 'log ', 
    'datetime_format': '%Y%m%d_%H%M%S',
    'save_videos_in_log_dir': True, # Сохранение видео в папку с логами
    'print_status': True, # Вывод статуса логирования
    'local_unit_count_threshold': 0.001, # Порог для оценки наличия движения в условных единицах
    
    'csv': {
        'settings': True, # 1: настройки, пороги, калибровки
        'keyframe_points': True, # 2: 3D-точки ключевых кадров
        'motion_points': True, # 3: 3D-точки, участвующие в оценке движения
        'frame_motion': True, # 4: перемещения/повороты/статистика по кадрам
        'scale_summary': True, # 5: сводка совпадений и масштаба слева/справа
        'scale_points': True, # 6: поточечные масштабы с соседними камерами
        'local_unit_motion_counts': True, # 7: счётчики приведённых локальных условных перемещений
        'global_metric_motion': True, # 8: глобальные метрические перемещения и повороты
    },

    'filenames': {
        'settings': '01_settings.csv',
        'keyframe_points': '02_keyframe_points.csv',
        'motion_points': '03_motion_points.csv',
        'frame_motion': '04_frame_motion.csv',
        'scale_summary': '05_scale_summary.csv',
        'scale_points': '06_scale_points.csv',
        'local_unit_motion_counts': '07_local_unit_motion_counts.csv',
        'local_unit_z_motion': '08_local_unit_z_motion.csv',
    }
}

# ------------------- НАСТРОЙКИ ОТОБРАЖЕНИЯ -------------------
DISPLAY_CONFIG = {
    'video_panel_width': 150, # Ширина панели для видео
    'kf_panel_width': 150, # Ширина панели для ключевых кадров
    'window_video': "Все видео", # Имя окна для отображения всех видео
    'window_kf': "Все репроекции", # Имя окна для репроекций ключевых кадров
    'summary_col_width': 150,# Ширина общей панели
    'font_scale': 0.3,# Масштаб шрифта панели
    'font_thickness': 1, # Толщина шрифта 
    'line_height': 14, # Высота строки текста
    'section_gap': 6, # Отступ между секциями
    'text_margin_x': 10, # Отступ текста от края
    'text_y_start': 18, # Начальная позиция текста по Y
    'camera_label_font_scale': 0.6, # Масштаб шрифта для меток камер
    'show_tracking_points': True, # Отображение точек отслеживания перемещений на видео
    'show_cross_camera_points': True, # Отображение точек найденных между кадрами
    'show_cross_camera_coords': False, # Отображение координат точек найденных между кадрами
    'cross_visual_dedupe_radius_px': 5.0, # Радиус для удаления дублирующихся точек при визуализации
}

# ------------------- НАСТРОЙКИ МЕТРИЧЕСКОГО МАСШТАБА -------------------
METRIC_SCALE_CONFIG = {
    'calib_unit_to_meter': 0.001, # Коэффициент перевода калибровочных единиц в метры (1 единица = 0.001 м)
    'min_scale_points': 6, # Минимальное количество точек для оценки метрического масштаба
    'grid_search_radius_px': 10.0, # Радиус поиска в пикселях для сопоставления стерео-точек
    'hamming_threshold': 200, # Порог Хэмминга для сопоставления дескрипторов
    'scale_reproj_error_px': 5.0, # Максимальная ошибка репроекции для точек масштаба
    'min_depth': 1e-6, # Минимальная глубина точек
}

num = 3

# ------------------- КОНФИГУРАЦИИ КАМЕР -------------------
CAMERA_CONFIGS = [
    {
        'video_path': f'/home/art/diplom/actual/multi_track/video_USB/dreif/izm_{num}/left_seg{num}.mkv', # Путь к видео
        'placement': 'left', # Размещение камеры
        'mapp_coeff': [2.886186263863493e+02, -0.001527614435019, 1.420904260140486e-06, -4.927110535062812e-09], # Коэффициенты калибровки a0 a2 a3 a4
        'img_size': (368, 640), # Размер кадра
        'dist_cnt': (3.186418515411832e+02, 1.831550588820148e+02), # Центр изображения
        'stretch_matrix': [[1.0, 0.0], [0.0, 1.0]], # Матрица афииных преобразований
        'nfeatures': 2000, # Максимальноее число точек ORB для обнаружения
        'scaleFactor': 1.2, # Коэффициент масштаба между уровнями пирамиды
        'nlevels': 8, # Число уровней пирамиды
        'score_threshold': 3000.0, # Минимальный score для инициализации 
        'min_parallax_deg': 1.8, # Минимальный параллакс для инициализации
        'min_matches_init': 200, # Минимальное число совпадений для инициализации
        'min_inliers_init': 150, # Минимальное число хороших точек для инициализации
        'normalize_init_scale': True, # Включение приведения к единичному масштабу после инициализации
        'homo_ransac_threshold': 0.008, # Порог RANSAC для гомографии
        'ess_ransac_threshold': 0.008, #  Порог RANSAC для существенной матрицы
        'ess_prob': 0.999, # Доверительная вероятность для RANSAC существенной матрицы
        'search_radius': 10, # Радиус поиска соответствий в пикселях
        'hamming_threshold': 250, # Максимальное расстояние Хэмминга для сопоставления дескрипторов
        'min_matches_kf': 200, # Количество совпадений, при котором создается новый ключевой кадр
        'kf_min_matches': 100, # Минимальное количество совпадений для создания ключевого кадра
        'kf_max_reproj_error': 5.0, # Максимальная ошибка репроекции в пикселях при создании ключевого кадра
    },
    {
        'video_path': f'/home/art/diplom/actual/multi_track/video_USB/dreif/izm_{num}/forward_seg{num}.mkv',
        'placement': 'front',
        'mapp_coeff': [2.843939401565993e+02, -0.001553980307285, 1.842296269190585e-06, -5.872654146970764e-09],
        'img_size': (368, 640),
        'dist_cnt': (3.214145963778078e+02, 1.912038587123719e+02),
        'stretch_matrix': [[1.0, 0.0], [0.0, 1.0]],
        'nfeatures': 2000,
        'scaleFactor': 1.2,
        'nlevels': 8,
        'score_threshold': 3000.0,
        'min_parallax_deg': 1.8,
        'min_matches_init': 200,
        'min_inliers_init': 150,
        'normalize_init_scale': True,
        'homo_ransac_threshold': 0.008,
        'ess_ransac_threshold': 0.008,
        'ess_prob': 0.999,
        'search_radius': 10,
        'hamming_threshold': 250,
        'min_matches_kf': 200,
        'kf_min_matches': 100,
        'kf_max_reproj_error': 5.0,
    },
    {
        'video_path': f'/home/art/diplom/actual/multi_track/video_USB/dreif/izm_{num}/back_seg{num}.mkv',
        'placement': 'back',
        'mapp_coeff': [2.888970736457522e+02, -0.001617897444982, 2.127701436878351e-06, -6.137436758193882e-09],
        'img_size': (368, 640),
        'dist_cnt': (3.407892834066074e+02, 2.081579836212685e+02),
        'stretch_matrix': [[1.0, 0.0], [0.0, 1.0]],
        'nfeatures': 2000,
        'scaleFactor': 1.2,
        'nlevels': 8,
        'score_threshold': 3000.0,
        'min_parallax_deg': 1.8,
        'min_matches_init': 200,
        'min_inliers_init': 150,
        'normalize_init_scale': True,
        'homo_ransac_threshold': 0.008,
        'ess_ransac_threshold': 0.008,
        'ess_prob': 0.999,
        'search_radius': 10,
        'hamming_threshold': 250,
        'min_matches_kf': 200,
        'kf_min_matches': 100,
        'kf_max_reproj_error': 5.0,
    },
    {
        'video_path': f'/home/art/diplom/actual/multi_track/video_USB/dreif/izm_{num}/right_seg{num}.mkv',
        'placement': 'right',
        'mapp_coeff': [2.845357304400566e+02, -0.001474898296301, 1.064536660784384e-06, -4.424709422779739e-09],
        'img_size': (368, 640),
        'dist_cnt': (3.270418722295847e+02, 1.987343710529728e+02),
        'stretch_matrix': [[1.0, 0.0], [0.0, 1.0]],
        'nfeatures': 2000,
        'scaleFactor': 1.2,
        'nlevels': 8,
        'score_threshold': 3000.0,
        'min_parallax_deg': 1.8,
        'min_matches_init': 200,
        'min_inliers_init': 150,
        'normalize_init_scale': True,
        'homo_ransac_threshold': 0.008,
        'ess_ransac_threshold': 0.008,
        'ess_prob': 0.999,
        'search_radius': 10,
        'hamming_threshold': 250,
        'min_matches_kf': 200,
        'kf_min_matches': 100,
        'kf_max_reproj_error': 5.0,
    }
]

# ------------------- КАЛИБРОВКА СТЕРЕОПАР -------------------
CROSS_CALIB = {
    # F-R
    (1, 3): {
        'R': np.array([[0.072760451794739, 0.000144883848255, -0.997349435084462],
                       [0.065826880455171, 0.997818794313350, 0.004947274461052],
                       [0.995174727605241, -0.066012367960276, 0.072592209031762]], dtype=np.float64),
        't': np.array([-1.390244639990861e+02, -4.983970375947052, -1.205441303756609e+02], dtype=np.float64)
    },
    # R-B
    (3, 2): {
        'R': np.array([[-0.134954319481673, -0.045280696223564, -0.989816644739190],
                       [0.075846423259059, 0.995552247448388, -0.055884189886519],
                       [0.987944660257874, -0.082615865001670, -0.130919697211610]], dtype=np.float64),
        't': np.array([-1.217647612309362e+02, -10.090875740971683, -75.335387819737335], dtype=np.float64)
    },
    # B-L
    (2, 0): {
        'R': np.array([[0.079768231050132, -0.101333738113307, -0.991649385032390],
                       [0.056029633776562, 0.993702467948612, -0.097036515455274],
                       [0.995237514090982, -0.047821320692981, 0.084943580294110]], dtype=np.float64),
        't': np.array([-79.783916406528945, -4.092343227669152, -1.179151975238433e+02], dtype=np.float64)
    },
    # L-F
    (0, 1): {
        'R': np.array([[0.000966331222545, -0.030960941895542, -0.999520128001887],
                       [0.006861867404951, 0.999497268348658, -0.030953599779152],
                       [0.999975990201607, -0.006828663157012, 0.001178295295273]], dtype=np.float64),
        't': np.array([-1.100668379654395e+02, -4.485401777458339, -1.582452743493218e+02], dtype=np.float64)
    }
}

CONFIGS = CAMERA_CONFIGS
