import numpy as np

APP_CONFIG = {
    'max_cams': 4,
    'default_frame_size': (480, 640),
    'anchor_camera_id': 1,
}

ORB_DEFAULTS = {
    'nfeatures': 2000,
    'scaleFactor': 1.2,
    'nlevels': 8,
    'edgeThreshold': 31,
    'firstLevel': 0,
    'WTA_K': 2,
    'scoreType': 'HARRIS',
    'patchSize': 31,
    'fastThreshold': 20,
}

TRACKER_DEFAULTS = {
    'placement': 'front',
    'nfeatures': ORB_DEFAULTS['nfeatures'],
    'scaleFactor': ORB_DEFAULTS['scaleFactor'],
    'nlevels': ORB_DEFAULTS['nlevels'],
    'score_threshold': 5000.0,
    'min_parallax_deg': 4.0,
    'min_matches_init': 400,
    'min_inliers_init': 400,
    'normalize_init_scale': True,
    'homo_ransac_threshold': 0.05,
    'ess_ransac_threshold': 0.008,
    'ess_prob': 0.999,
    'search_radius': 10,
    'hamming_threshold': 300,
    'min_matches_kf': 200,
    'kf_min_matches': 100,
    'kf_max_reproj_error': 5.0,
    'metric_scale_alpha': 0.15,
    'metric_scale_max_jump': 1.2,

    'point_follow_filter_enabled': False,

    'point_follow_prev_frames': 3,

    'point_follow_max_prediction_error_px': 20.0,
    'point_follow_prediction_error_factor': 4.0,
    'point_follow_max_step_px': 2.0,
    'point_follow_step_factor': 4.0,
    'point_follow_min_motion_px': 1.0,
}

TRACKER_DISPLAY_DEFAULTS = {
    'panel_width': 220,
    'font_scale': 0.5,
    'font_thickness': 1,
    'display_margin_x': 10,
    'display_y_start': 24,
    'display_line_height': 20,
    'display_section_gap': 8,
}


MODEL_ESTIMATION_DEFAULTS = {
    'homography_ransac_threshold': 1.0,
    'essential_ransac_threshold': 1.0,
    'essential_prob': 0.999,
}

EVALUATE_PAIR_CONFIG = {
    'homo_thresh': 0.05,
    'ess_thresh': 0.008,
    'ess_prob': 0.999,
    'min_matches': 500,
}

POSE_MATCH_CONFIG = {
    'search_radius': 10,
    'hamming_threshold': 200,
    'visualize': True,
    'panel_width': 220,
}

PNP_CONFIG = {
    'min_correspondences': 4,
    'ransac_iterations_count': 200,
    'ransac_reprojection_error_norm': 0.02,
    'ransac_confidence': 0.99,
    'refine_lm_max_iter': 30,
    'refine_lm_eps': 1e-7,
    'ba_iterations': 3,
    'ba_refine_max_iter': 20,
    'ba_refine_eps': 1e-6,
    'final_pixel_error_min': 4.0,
    'final_pixel_error_radius_factor': 0.5,
}

KEYFRAME_CONFIG = {
    'min_matches': 100,
    'max_reproj_error': 5.0,
    'min_positive_depth_points': 10,
    'min_good_points': 10,
    'min_parallax_deg': 0.5,
    'min_baseline_unit': 0.04,
    'min_baseline_rel': 0.04,
}

CROSS_PAIRS = [
    (1, 3, (255, 0, 0), "F-R"),
    (3, 2, (0, 255, 0), "R-B"),
    (2, 0, (0, 0, 255), "B-L"),
    (0, 1, (255, 255, 0), "L-F"),
]

CROSS_MATCH_CONFIG = {
    'hamming_threshold': 200,
}

GEOMETRY_CONFIG = {
    'use_geometric_filter': True,
    'epipolar_threshold': 7.0,
}

RECORDING_CONFIG = {
    'save_final_window': True,
    'save_keyframe_reprojection': True,
    'final_window_path': 'outputs/final_window.avi',
    'keyframe_reprojection_path': 'outputs/keyframe_reprojection.avi',
    'fps': None,
    'fallback_fps': 30.0,
    'fourcc': 'XVID',
    'keyframe_grid_cols': 2,
    'resize_on_size_change': True,
    'print_status': True,
}

LOGGING_CONFIG = {
    'enabled': True,

    'root_dir': 'outputs',
    'directory_prefix': 'log',
    'datetime_format': '%Y%m%d_%H%M%S',

    'save_videos_in_log_dir': True,

    'print_status': True,

    'local_unit_count_threshold': 0.01,
    
    'csv': {
        'settings': True,    
        'keyframe_points': True,
        'motion_points': True,
        'frame_motion': True,
        'scale_summary': True,
        'scale_points': True,
        'local_unit_motion_counts': True,
        'global_metric_motion': True,
    },

    'filenames': {
        'settings': '01_settings.csv',
        'keyframe_points': '02_keyframe_points.csv',
        'motion_points': '03_motion_points.csv',
        'frame_motion': '04_frame_motion.csv',
        'scale_summary': '05_scale_summary.csv',
        'scale_points': '06_scale_points.csv',
        'local_unit_motion_counts': '07_local_unit_motion_counts.csv',
        'global_metric_motion': '08_global_metric_motion.csv',
    }
}

DISPLAY_CONFIG = {
    'video_panel_width': 150,
    'kf_panel_width': 150,
    'window_video': "Все видео",
    'window_kf': "Все репроекции",
    'summary_col_width': 150,
    'font_scale': 0.4,
    'font_thickness': 1,
    'line_height': 18,
    'section_gap': 6,
    'text_margin_x': 10,
    'text_y_start': 22,
    'camera_label_font_scale': 0.6,
    'show_tracking_points': True,
    'show_cross_camera_points': True,
    'show_cross_camera_coords': False,
    'cross_visual_dedupe_radius_px': 5.0
}

METRIC_SCALE_CONFIG = {
    'calib_unit_to_meter': 0.001,

    'min_scale_points': 6,

    'grid_search_radius_px': 10.0,
    'hamming_threshold': 200,


    'scale_reproj_error_px': 5.0,


    'min_depth': 1e-6,
}

CAMERA_CONFIGS = [
    {
        'video_path': '/home/art/diplom/actual/multi_track/video_ABCD/rotate_step5minstep_full90/cam_A.avi',
        'placement': 'left',
        'mapp_coeff': [3.131696721475739e+02, -0.001330630663917, 1.396946031095517e-06, -4.522658697877816e-09],
        'img_size': (480, 640),
        'dist_cnt': (3.530917809918790e+02, 2.402092440693070e+02),
        'stretch_matrix': [[1.0, 0.0], [0.0, 1.0]],
        'nfeatures': 2000,
        'scaleFactor': 1.2,
        'nlevels': 8,
        'score_threshold': 5000.0,
        'min_parallax_deg': 2.5,
        'min_matches_init': 200,
        'min_inliers_init': 200,
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
        'video_path': '/home/art/diplom/actual/multi_track/video_ABCD/rotate_step5minstep_full90/cam_B.avi',
        'placement': 'front',
        'mapp_coeff': [3.108481958935834e+02, -0.001377160549228, 1.562283919512581e-06, -4.675406570447146e-09],
        'img_size': (480, 640),
        'dist_cnt': (3.254690081398768e+02, 2.422602761545476e+02),
        'stretch_matrix': [[1.0, 0.0], [0.0, 1.0]],
        'nfeatures': 2000,
        'scaleFactor': 1.2,
        'nlevels': 8,
        'score_threshold': 5000.0,
        'min_parallax_deg': 2.5,
        'min_matches_init': 200,
        'min_inliers_init': 200,
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
        'video_path': '/home/art/diplom/actual/multi_track/video_ABCD/rotate_step5minstep_full90/cam_C.avi',
        'placement': 'back',
        'mapp_coeff': [3.133325874742267e+02, -0.001324633807446, 1.324725487316811e-06, -4.344618843619976e-09],
        'img_size': (480, 640),
        'dist_cnt': (3.210352663469159e+02, 2.142699103447184e+02),
        'stretch_matrix': [[1.0, 0.0], [0.0, 1.0]],
        'nfeatures': 2000,
        'scaleFactor': 1.2,
        'nlevels': 8,
        'score_threshold': 5000.0,
        'min_parallax_deg': 2.5,
        'min_matches_init': 200,
        'min_inliers_init': 200,
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
        'video_path': '/home/art/diplom/actual/multi_track/video_ABCD/rotate_step5minstep_full90/cam_D.avi',
        'placement': 'right',
        'mapp_coeff': [3.124899259293122e+02, -0.001339465010499, 1.433964473685873e-06, -4.474942507213656e-09],
        'img_size': (480, 640),
        'dist_cnt': (3.366810748863500e+02, 2.194111704414140e+02),
        'stretch_matrix': [[1.0, 0.0], [0.0, 1.0]],
        'nfeatures': 2000,
        'scaleFactor': 1.2,
        'nlevels': 8,
        'score_threshold': 5000.0,
        'min_parallax_deg': 2.5,
        'min_matches_init': 200,
        'min_inliers_init': 200,
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

CROSS_CALIB = {
    (1, 3): {
        'R': np.array([[-0.007465277994812, -0.041790370658630, -0.999098510931067],
                       [0.014411058776250, 0.999018096238191, -0.041894686698542],
                       [0.999868286830525, -0.014710822847001, -0.006855704322423]], dtype=np.float64),
        't': np.array([-1.301213985203270e+02, 0.319034518302158, -1.091542206878426e+02], dtype=np.float64)
    },
    (3, 2): {
        'R': np.array([[0.003700803054113, -0.000936134454116, -0.999992713827975],
                       [-0.008292046834458, 0.999965153025133, -0.000966796109155],
                       [0.999958772158158, 0.008295564339171, 0.003692911621585]], dtype=np.float64),
        't': np.array([-1.301923357804900e+02, -6.238977373634792, -1.154620463243421e+02], dtype=np.float64)
    },
    (2, 0): {
        'R': np.array([[0.005478806642606, 0.020598174165798, -0.999772823144742],
                       [-0.003510189427539, 0.999782065186443, 0.020579128560855],
                       [0.999978830399902, 0.003396642927485, 0.005549916113549]], dtype=np.float64),
        't': np.array([-1.506604840908259e+02, -2.439567092847661, -1.341693820258484e+02], dtype=np.float64)
    },
    (0, 1): {
        'R': np.array([[-0.010625761660238, 0.019108660363352, -0.999760947571097],
                       [0.047633315582432, 0.998692033455037, 0.018581968681159],
                       [0.998808370227124, -0.047424481152261, -0.011522072368917]], dtype=np.float64),
        't': np.array([-1.191585439650334e+02, -1.741260472523966, -1.429333285066866e+02], dtype=np.float64)
    }
}

CONFIGS = CAMERA_CONFIGS
