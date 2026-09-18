"""棋盘格检测方向、尺度及图像到手眼求解的合成验证，不访问硬件。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from piper_capture.config import load_config
from piper_capture.handeye import build_detector, detect_board, session_dir, solve
from piper_capture.jsonio import write_json
from piper_capture.schema import Transform


class CheckerboardTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/handeye_checkerboard_eye_in_hand.json')
        self.board = self.cfg['handeye']['board']
        self.K = np.array([[1000., 0, 640], [0, 1000., 480], [0, 0, 1]])
        self.camera = SimpleNamespace(color_intrinsics={
            'fx': 1000., 'fy': 1000., 'ppx': 640., 'ppy': 480., 'coeffs': [0.] * 5})

    def render(self, rotation, distance=.45):
        R = cv2.Rodrigues(np.asarray(rotation, dtype=float))[0]
        t = np.array([0., 0., distance]) - R @ np.array([.0675, .045, 0.])
        T = Transform.from_rt(R, t)
        # 2× 超采样，降低栅格化角点误差。
        img = np.full((1920, 2560, 3), 255, np.uint8)
        K = self.K.copy(); K[:2] *= 2
        for row in range(8):
            for col in range(11):
                if (row + col) % 2:
                    continue
                p = np.array([[col-1, row-1, 0], [col, row-1, 0],
                              [col, row, 0], [col-1, row, 0]], dtype=float) * .015
                pixels = cv2.projectPoints(p, np.asarray(rotation, dtype=float), t, K, None)[0]
                cv2.fillConvexPoly(img, np.rint(pixels.reshape(-1, 2) * 16).astype(np.int32),
                                   (0, 0, 0), shift=4)
        return cv2.resize(img, (1280, 960), interpolation=cv2.INTER_AREA), T

    def test_fixed_board_origin_across_rotations(self):
        for angle in [0., np.pi/2, np.pi, -np.pi/2]:
            with self.subTest(angle=angle):
                img, truth = self.render([.1, -.15, angle])
                det = detect_board(img, self.board, self.camera)
                self.assertTrue(det.get('quality_ok'), det)
                actual = np.asarray(det['T_cam_target'])
                self.assertLess(np.linalg.norm(actual[:3, 3] - truth[:3, 3]), .001)
                self.assertLess(Transform.rotation_angle_deg(actual[:3, :3], truth[:3, :3]), .5)

    def test_blank_wrong_shape_and_invalid_config(self):
        blank = np.full((960, 1280, 3), 255, np.uint8)
        self.assertFalse(detect_board(blank, self.board, self.camera)['found'])
        img, _ = self.render([.1, -.15, .2])
        wrong = {**self.board, 'inner_corners': [9, 8]}
        self.assertFalse(detect_board(img, wrong, self.camera)['found'])
        for patch in [{'inner_corners': [9, 9]}, {'square_size_m': 0}, {'square_size_m': float('nan')}]:
            with self.assertRaises(ValueError):
                build_detector({**self.board, **patch})

    def test_scale_and_rendered_handeye(self):
        img, _ = self.render([.2, -.15, .1])
        normal = detect_board(img, self.board, self.camera)
        doubled = detect_board(img, {**self.board, 'square_size_m': .03}, self.camera)
        np.testing.assert_allclose(doubled['tvec_m'], np.asarray(normal['tvec_m'])*2, atol=1e-6)
        X = Transform.from_rt(cv2.Rodrigues(np.array([.2, -.3, .1]))[0], [.02, .075, .03])
        Y = Transform.from_rt(np.eye(3), [.4, .1, .05])
        rng = np.random.default_rng(582)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); d = session_dir(root, 'synthetic-checker')
            write_json(d/'session.json', {'board': self.board, 'camera': {'calibration_id': 'synthetic'}})
            for i in range(24):
                img, B = self.render(rng.uniform([-.5, -.5, -.8], [.5, .5, .8]), rng.uniform(.35, .55))
                det = detect_board(img, self.board, self.camera)
                self.assertTrue(det.get('quality_ok'), det)
                A = Y @ np.linalg.inv(B) @ np.linalg.inv(X)
                write_json(d/f'sample_{i:04d}.json', {
                    'sample_index': i, 'robot': {'T_base_ee': A.tolist()}, 'detection': det})
            result = solve(self.cfg, root, session_id='synthetic-checker', save=False)
            self.assertEqual(result['status'], 'valid')
            actual = np.asarray(result['transform']['matrix_row_major'])
            self.assertLess(np.linalg.norm(actual[:3, 3]-X[:3, 3]), .003)
            self.assertLess(Transform.rotation_angle_deg(actual[:3, :3], X[:3, :3]), .5)
            # 跨会话验证必须拒绝不同的格长或角点布局。
            other = session_dir(root, 'different-board')
            write_json(other/'session.json', {'board': {**self.board, 'square_size_m': .02},
                                             'camera': {'calibration_id': 'synthetic'}})
            mismatch = solve(self.cfg, root, session_id='synthetic-checker',
                             verify_session_id='different-board', save=False)
            self.assertEqual(mismatch['status'], 'invalid')
            self.assertIn('标定板配置不一致', mismatch['reason'])


if __name__ == '__main__':
    unittest.main()
