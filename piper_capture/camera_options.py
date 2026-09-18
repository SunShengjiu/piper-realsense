"""RealSense imaging controls shared by the tuner and capture startup."""
from __future__ import annotations

import math
from typing import Any, Dict

# Imaging controls only: transport, synchronization, depth units and firmware
# operations are deliberately not exposed as image adjustments.
LABELS = {
    'enable_auto_exposure': '自动曝光', 'exposure': '曝光', 'gain': '增益',
    'enable_auto_white_balance': '自动白平衡', 'white_balance': '白平衡（K）',
    'brightness': '亮度', 'contrast': '对比度', 'saturation': '饱和度',
    'sharpness': '锐度', 'gamma': '伽马', 'hue': '色调',
    'backlight_compensation': '背光补偿', 'power_line_frequency': '电源防闪烁',
    'visual_preset': '深度预设', 'emitter_enabled': '红外发射器',
    'laser_power': '激光功率', 'auto_exposure_priority': '自动曝光优先',
    'auto_exposure_mode': '自动曝光模式', 'auto_exposure_converge_step': '自动曝光收敛步长',
    'auto_exposure_limit': '自动曝光上限', 'auto_gain_limit': '自动增益上限',
    'auto_exposure_limit_toggle': '启用自动曝光上限',
    'auto_gain_limit_toggle': '启用自动增益上限',
    'emitter_on_off': '交替发射', 'emitter_always_on': '发射器常开',
}
AUTO_FOR = {'exposure': 'enable_auto_exposure', 'gain': 'enable_auto_exposure',
            'white_balance': 'enable_auto_white_balance'}
DEPTH_LABELS = {'enable_auto_exposure': '红外自动曝光', 'exposure': '红外曝光（μs）',
                'gain': '红外增益', 'laser_power': '投射器功率（mW）',
                'auto_exposure_limit': '红外自动曝光上限（μs）'}


def imaging_sensors(device: Any, rs: Any) -> Dict[str, Any]:
    result = {}
    for sensor in device.query_sensors():
        types = {p.stream_type() for p in sensor.get_stream_profiles()}
        if rs.stream.color in types:
            result['color'] = sensor
        if rs.stream.depth in types:
            result['depth'] = sensor
    return result


def describe_options(device: Any, rs: Any) -> Dict[str, Any]:
    result = {}
    for kind, sensor in imaging_sensors(device, rs).items():
        controls = []
        for name, label in LABELS.items():
            option = getattr(rs.option, name, None)
            if option is None or not sensor.supports(option):
                continue
            try:
                bounds = sensor.get_option_range(option)
                choices = {}
                if bounds.step == 1 and bounds.max - bounds.min <= 16:
                    for val in range(int(bounds.min), int(bounds.max) + 1):
                        try:
                            desc = sensor.get_option_value_description(option, float(val))
                        except RuntimeError:
                            desc = None
                        if desc:
                            choices[str(val)] = desc
                controls.append(dict(name=name, label=DEPTH_LABELS.get(name, label) if kind == 'depth' else label,
                                     value=sensor.get_option(option),
                                     min=bounds.min, max=bounds.max, step=bounds.step,
                                     default=bounds.default, readonly=sensor.is_option_read_only(option),
                                     description=sensor.get_option_description(option), choices=choices))
            except RuntimeError:
                continue
        result[kind] = controls
    return result


def saved_options(device: Any, rs: Any) -> Dict[str, Dict[str, float]]:
    """Save stable controls; an AE/AWB measurement is not a manual setpoint."""
    result = {}
    for kind, controls in describe_options(device, rs).items():
        values = {c['name']: c['value'] for c in controls if not c['readonly']}
        for manual, auto in AUTO_FOR.items():
            if values.get(auto):
                values.pop(manual, None)
        result[kind] = values
    return result


def set_option(device: Any, rs: Any, kind: str, name: str, value: float) -> float:
    if name not in LABELS:
        raise ValueError(f'不支持的图像参数: {name}')
    sensor = imaging_sensors(device, rs).get(kind)
    option = getattr(rs.option, name, None)
    if sensor is None or option is None or not sensor.supports(option):
        raise ValueError(f'{kind}.{name}: 此设备不支持')
    if sensor.is_option_read_only(option):
        raise ValueError(f'{kind}.{name}: 当前为只读')
    value = float(value)
    bounds = sensor.get_option_range(option)
    if not math.isfinite(value) or not bounds.min <= value <= bounds.max:
        raise ValueError(f'{name}: 允许范围 {bounds.min} ~ {bounds.max}')
    if bounds.step > 0:
        steps = (value - bounds.min) / bounds.step
        if abs(steps - round(steps)) > 1e-3:
            raise ValueError(f'{name}: 步长必须为 {bounds.step}')
    # Some D435i firmware rejects an unchanged limit toggle during streaming
    # with "No expected user action". A matching readback already satisfies it.
    if name in ('auto_exposure_limit_toggle', 'auto_gain_limit_toggle'):
        if float(sensor.get_option(option)) == value:
            return value
    auto = AUTO_FOR.get(name)
    if auto and sensor.supports(getattr(rs.option, auto)):
        sensor.set_option(getattr(rs.option, auto), 0)
    sensor.set_option(option, value)
    return float(sensor.get_option(option))


def apply_options(device: Any, rs: Any, settings: Dict[str, Dict[str, float]]) -> None:
    """Apply preset first, auto modes next, manual values only when auto is off."""
    for kind, values in settings.items():
        if kind not in ('color', 'depth') or not isinstance(values, dict):
            raise ValueError(f'非法 sensor_options 分组: {kind}')
        priority = lambda name: (0 if name == 'visual_preset' else 1 if name.startswith('enable_auto') else 2, name)
        for name in sorted(values, key=priority):
            if values.get(AUTO_FOR.get(name, '')):
                continue
            try:
                set_option(device, rs, kind, name, values[name])
            except Exception as exc:
                raise RuntimeError(f'应用相机参数 {kind}.{name}={values[name]} 失败: {exc}') from exc
