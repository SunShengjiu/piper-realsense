"""Persistence and hardware-setting behavior, without requiring USB devices."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from piper_capture.camera_options import apply_options, saved_options, set_option
from piper_capture.camera_tuner import infrared_stats, save_configuration, validate_specs
from piper_capture.config import load_config
from piper_capture.lerobot_writer import LeRobotCaptureSession, make_features


@pytest.fixture
def config():
    return load_config('configs/lerobot_v3_two_d435i.json')


def test_save_preserves_capture_settings_and_reloads_options(tmp_path, config):
    specs = {r: copy.deepcopy(config['camera'][r]) for r in ('wrist', 'third_person')}
    specs['wrist']['sensor_options'] = {'color': {'enable_auto_exposure': 0, 'exposure': 200}}
    specs['third_person']['sensor_options'] = {'color': {'enable_auto_exposure': 1, 'brightness': 12}}
    output = tmp_path / 'tuned.json'
    save_configuration(config, specs, output)
    restored = load_config(output)
    assert restored['robot'] == config['robot']
    assert restored['lerobot'] == config['lerobot']
    session = LeRobotCaptureSession(restored, base_dir=tmp_path)
    for role in specs:
        camera = session._camera(restored['camera'][role])
        assert camera.serial == specs[role]['serial']
        assert camera.sensor_options == specs[role]['sensor_options']
    assert config['camera']['wrist'].get('sensor_options') is None


def test_invalid_save_does_not_replace_existing_file(tmp_path, config):
    specs = {r: copy.deepcopy(config['camera'][r]) for r in ('wrist', 'third_person')}
    output = tmp_path / 'existing.json'
    output.write_text('{"preserved":true}')
    specs['third_person']['serial'] = specs['wrist']['serial']
    with pytest.raises(ValueError, match='序列号'):
        save_configuration(config, specs, output)
    assert json.loads(output.read_text()) == {'preserved': True}


def test_auto_values_not_saved_as_manual_setpoints():
    controls = {'color': [dict(name=n, value=v, readonly=False) for n, v in [
        ('enable_auto_exposure', 1), ('exposure', 700), ('gain', 10),
        ('enable_auto_white_balance', 1), ('white_balance', 4500), ('brightness', 20)]]}
    with patch('piper_capture.camera_options.describe_options', return_value=controls):
        result = saved_options(None, None)
    assert result['color'] == {'enable_auto_exposure': 1, 'enable_auto_white_balance': 1, 'brightness': 20}


def test_preset_and_auto_modes_precede_manual_settings():
    settings = {'depth': {'exposure': 500, 'visual_preset': 1, 'enable_auto_exposure': 0},
                'color': {'enable_auto_exposure': 1, 'exposure': 20, 'gain': 1}}
    with patch('piper_capture.camera_options.set_option') as setter:
        apply_options(None, None, settings)
    assert [(c.args[2], c.args[3]) for c in setter.call_args_list] == [
        ('depth', 'visual_preset'), ('depth', 'enable_auto_exposure'), ('depth', 'exposure'),
        ('color', 'enable_auto_exposure')]


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 101, 1.5])
def test_invalid_manual_value_never_disables_auto(value):
    from unittest.mock import Mock
    sensor = Mock()
    sensor.supports.return_value = True
    sensor.is_option_read_only.return_value = False
    sensor.get_option_range.return_value = SimpleNamespace(min=0, max=100, step=1)
    rs = SimpleNamespace(option=SimpleNamespace(exposure='exposure', enable_auto_exposure='auto'))
    with patch('piper_capture.camera_options.imaging_sensors', return_value={'color': sensor}):
        with pytest.raises(ValueError):
            set_option(None, rs, 'color', 'exposure', value)
    sensor.set_option.assert_not_called()


def test_independent_camera_resolutions():
    features = make_features(720, 1280, .001, .001, third_height=480, third_width=640)
    assert features['observation.images.wrist']['shape'] == (720, 1280, 3)
    assert features['observation.images.third_person']['shape'] == (480, 640, 3)
    assert features['observation.images.third_person_depth']['shape'] == (480, 640, 1)


def test_infrared_stats_preserve_absolute_brightness_and_missing_frames():
    import numpy as np
    assert infrared_stats(None) is None
    assert infrared_stats(np.empty((0, 0), dtype=np.uint8)) is None
    values = np.array([[0, 5, 6], [249, 250, 255]], dtype=np.uint8)
    assert infrared_stats(values) == dict(mean=127.5, dark_percent=33.3, saturated_percent=33.3)
    assert infrared_stats(np.full((4, 4), 255, dtype=np.uint8))['saturated_percent'] == 100
    assert infrared_stats(np.full((4, 4), 80, dtype=np.uint8)) == dict(
        mean=80.0, dark_percent=0.0, saturated_percent=0.0)


@pytest.mark.parametrize('name', ['auto_exposure_limit_toggle', 'auto_gain_limit_toggle'])
@pytest.mark.parametrize('value', [0, 1])
def test_limit_toggle_skips_only_unchanged_firmware_write(name, value):
    from unittest.mock import Mock
    sensor = Mock()
    sensor.supports.return_value = True
    sensor.is_option_read_only.return_value = False
    sensor.get_option_range.return_value = SimpleNamespace(min=0, max=1, step=1)
    sensor.get_option.return_value = 0
    rs = SimpleNamespace(option=SimpleNamespace(**{name: name}))
    with patch('piper_capture.camera_options.imaging_sensors', return_value={'depth': sensor}):
        set_option(None, rs, 'depth', name, value)
    if value == 0:
        sensor.set_option.assert_not_called()
    else:
        sensor.set_option.assert_called_once_with(name, 1.0)
