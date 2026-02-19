#!/usr/bin/env python3
"""
Angelus Paint Color Mixer
=========================
Angelus絵の具の配合比率を自動計算するエンジン

使用方法:
    python color_mixer.py "#FF5500"
    python color_mixer.py 255 85 0
"""

import sys
import numpy as np
from scipy.optimize import minimize

# Angelus基本色のRGB値
BASE_COLORS = {
    "001-Black": np.array([20, 20, 25]),
    "005-White": np.array([255, 255, 255]),
    "040-Blue": np.array([0, 90, 170]),
    "050-Green": np.array([0, 130, 70]),
    "064-Red": np.array([200, 35, 40]),
    "075-Yellow": np.array([255, 210, 0]),
}

# 表示用の色名（日本語）
COLOR_NAMES_JP = {
    "001-Black": "ブラック",
    "005-White": "ホワイト",
    "040-Blue": "ブルー",
    "050-Green": "グリーン",
    "064-Red": "レッド",
    "075-Yellow": "イエロー",
}

# 円グラフ用の色
CHART_COLORS = {
    "001-Black": "#141419",
    "005-White": "#FFFFFF",
    "040-Blue": "#005AAA",
    "050-Green": "#008246",
    "064-Red": "#C82328",
    "075-Yellow": "#FFD200",
}


def hex_to_rgb(hex_color: str) -> np.ndarray:
    """HEXカラーコードをRGBに変換"""
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)])


def rgb_to_hex(rgb: np.ndarray) -> str:
    """RGBをHEXカラーコードに変換"""
    return "#{:02X}{:02X}{:02X}".format(
        int(np.clip(rgb[0], 0, 255)),
        int(np.clip(rgb[1], 0, 255)),
        int(np.clip(rgb[2], 0, 255)),
    )


def blend_colors(ratios: dict) -> np.ndarray:
    """
    配合比率から混色結果のRGBを計算

    絵の具は減法混色だが、簡易的に加重平均で近似
    より正確にはCMYK変換が必要だが、実用上はこれで十分
    """
    result = np.zeros(3)
    for color_name, ratio in ratios.items():
        if ratio > 0:
            result += BASE_COLORS[color_name] * ratio
    return result


def calculate_color_difference(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    """2色間のユークリッド距離（色差）を計算"""
    return np.sqrt(np.sum((rgb1 - rgb2) ** 2))


def calculate_mix(target_rgb: np.ndarray, use_all_colors: bool = True) -> dict:
    """
    ターゲット色に最も近い配合比率を計算

    Args:
        target_rgb: 目標のRGB値 (numpy array)
        use_all_colors: 全色を使用するかどうか

    Returns:
        配合比率の辞書
    """
    color_names = list(BASE_COLORS.keys())
    n_colors = len(color_names)
    color_matrix = np.array([BASE_COLORS[name] for name in color_names])

    def objective(ratios):
        """目的関数: 色差の最小化"""
        blended = np.dot(ratios, color_matrix)
        return calculate_color_difference(blended, target_rgb)

    # 制約条件: 合計 = 1
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    # 境界条件: 各比率は0〜1
    bounds = [(0, 1) for _ in range(n_colors)]

    # 初期値: 均等配分
    x0 = np.ones(n_colors) / n_colors

    # 最適化実行
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 1000},
    )

    # 結果を辞書に変換（1%未満は0に）
    ratios = {}
    for i, name in enumerate(color_names):
        ratio = result.x[i]
        if ratio >= 0.01:  # 1%以上のみ
            ratios[name] = round(ratio, 4)

    # 正規化（合計を1に）
    total = sum(ratios.values())
    if total > 0:
        ratios = {k: round(v / total, 4) for k, v in ratios.items()}

    return ratios


def calculate_mix_result(target_rgb: np.ndarray) -> dict:
    """
    配合計算の完全な結果を返す

    Returns:
        {
            "target": {"rgb": [r,g,b], "hex": "#RRGGBB"},
            "ratios": {"001-Black": 0.2, ...},
            "result": {"rgb": [r,g,b], "hex": "#RRGGBB"},
            "error": 誤差率(%)
        }
    """
    ratios = calculate_mix(target_rgb)
    result_rgb = blend_colors(ratios)
    error = calculate_color_difference(target_rgb, result_rgb)
    error_percent = (error / 441.67) * 100  # 最大誤差(白→黒)で正規化

    return {
        "target": {"rgb": target_rgb.tolist(), "hex": rgb_to_hex(target_rgb)},
        "ratios": ratios,
        "ratios_percent": {k: round(v * 100, 1) for k, v in ratios.items()},
        "result": {"rgb": result_rgb.tolist(), "hex": rgb_to_hex(result_rgb)},
        "error": round(error_percent, 2),
        "color_names_jp": COLOR_NAMES_JP,
        "chart_colors": CHART_COLORS,
    }


def print_result(result: dict):
    """結果をコンソールに出力"""
    print("\n" + "=" * 50)
    print("Angelus Paint Color Mixer")
    print("=" * 50)
    print(f"\n目標色: {result['target']['hex']}")
    print(f"RGB: {result['target']['rgb']}")
    print("\n--- 配合比率 ---")

    for color_name, percent in sorted(
        result["ratios_percent"].items(), key=lambda x: -x[1]
    ):
        jp_name = COLOR_NAMES_JP.get(color_name, color_name)
        bar = "█" * int(percent / 5) + "░" * (20 - int(percent / 5))
        print(f"{color_name} ({jp_name}): {bar} {percent}%")

    print(f"\n再現色: {result['result']['hex']}")
    print(f"RGB: [" + ", ".join(f"{int(v)}" for v in result["result"]["rgb"]) + "]")
    print(f"誤差: {result['error']}%")
    print("=" * 50)


def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print('  python color_mixer.py "#FF5500"')
        print("  python color_mixer.py 255 85 0")
        sys.exit(1)

    # 引数をパース
    if len(sys.argv) == 2:
        # HEXカラー
        target_rgb = hex_to_rgb(sys.argv[1])
    elif len(sys.argv) == 4:
        # RGB値
        target_rgb = np.array([int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])])
    else:
        print("エラー: 引数が不正です")
        sys.exit(1)

    result = calculate_mix_result(target_rgb)
    print_result(result)


if __name__ == "__main__":
    main()
