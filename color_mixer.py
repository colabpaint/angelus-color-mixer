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
from scipy.optimize import minimize, differential_evolution

# Angelus基本色（軸となる6色）
PRIMARY_COLORS = {
    "001-Black": np.array([20, 20, 25]),
    "005-White": np.array([255, 255, 255]),
    "040-Blue": np.array([0, 90, 170]),
    "050-Green": np.array([0, 130, 70]),
    "064-Red": np.array([200, 35, 40]),
    "075-Yellow": np.array([255, 210, 0]),
}

# 補助色（より正確な配合のため）
SECONDARY_COLORS = {
    "024-Orange": np.array([255, 100, 30]),
    "047-Purple": np.array([100, 50, 120]),
    "014-Brown": np.array([90, 55, 40]),
    "029-Tan": np.array([210, 170, 130]),
    "041-Light Blue": np.array([80, 160, 210]),
    "052-Midnight Green": np.array([0, 75, 70]),
    "060-Burgundy": np.array([120, 30, 50]),
    "080-Dark Grey": np.array([80, 80, 85]),
    "082-Light Grey": np.array([180, 180, 185]),
    "188-Pink": np.array([240, 150, 170]),
    "186-Hot Pink": np.array([255, 80, 140]),
}

# 全色を結合
BASE_COLORS = {**PRIMARY_COLORS, **SECONDARY_COLORS}

# 表示用の色名（日本語）
COLOR_NAMES_JP = {
    "001-Black": "ブラック",
    "005-White": "ホワイト",
    "040-Blue": "ブルー",
    "050-Green": "グリーン",
    "064-Red": "レッド",
    "075-Yellow": "イエロー",
    "024-Orange": "オレンジ",
    "047-Purple": "パープル",
    "014-Brown": "ブラウン",
    "029-Tan": "タン",
    "041-Light Blue": "ライトブルー",
    "052-Midnight Green": "ミッドナイトグリーン",
    "060-Burgundy": "バーガンディ",
    "080-Dark Grey": "ダークグレー",
    "082-Light Grey": "ライトグレー",
    "188-Pink": "ピンク",
    "186-Hot Pink": "ホットピンク",
}

# 円グラフ用の色
CHART_COLORS = {
    "001-Black": "#141419",
    "005-White": "#FFFFFF",
    "040-Blue": "#005AAA",
    "050-Green": "#008246",
    "064-Red": "#C82328",
    "075-Yellow": "#FFD200",
    "024-Orange": "#FF641E",
    "047-Purple": "#643278",
    "014-Brown": "#5A3728",
    "029-Tan": "#D2AA82",
    "041-Light Blue": "#50A0D2",
    "052-Midnight Green": "#004B46",
    "060-Burgundy": "#781E32",
    "080-Dark Grey": "#505055",
    "082-Light Grey": "#B4B4B9",
    "188-Pink": "#F096AA",
    "186-Hot Pink": "#FF508C",
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


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGBをCIE Lab色空間に変換（人間の知覚に近い）"""
    # RGB to XYZ
    rgb_normalized = rgb / 255.0

    # sRGBガンマ補正
    mask = rgb_normalized > 0.04045
    rgb_linear = np.where(mask, ((rgb_normalized + 0.055) / 1.055) ** 2.4, rgb_normalized / 12.92)

    # RGB to XYZ matrix (sRGB, D65)
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = np.dot(matrix, rgb_linear * 100)

    # XYZ to Lab (D65 white point)
    ref = np.array([95.047, 100.000, 108.883])
    xyz_normalized = xyz / ref

    mask = xyz_normalized > 0.008856
    f = np.where(mask, xyz_normalized ** (1/3), (7.787 * xyz_normalized) + (16/116))

    L = (116 * f[1]) - 16
    a = 500 * (f[0] - f[1])
    b = 200 * (f[1] - f[2])

    return np.array([L, a, b])


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIE Lab色空間をRGBに変換"""
    L, a, b = lab

    # Lab to XYZ
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    def f_inv(t):
        return np.where(t > 0.206893, t ** 3, (t - 16/116) / 7.787)

    ref = np.array([95.047, 100.000, 108.883])
    xyz = np.array([f_inv(fx), f_inv(fy), f_inv(fz)]) * ref / 100

    # XYZ to RGB matrix
    matrix = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    rgb_linear = np.dot(matrix, xyz)

    # ガンマ補正
    mask = rgb_linear > 0.0031308
    rgb = np.where(mask, 1.055 * (rgb_linear ** (1/2.4)) - 0.055, 12.92 * rgb_linear)

    return np.clip(rgb * 255, 0, 255)


def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """CIE76色差（ΔE）を計算"""
    return np.sqrt(np.sum((lab1 - lab2) ** 2))


# RGB平均による混色計算
def blend_colors(ratios: dict) -> np.ndarray:
    """
    配合比率から混色結果のRGBを計算（RGB平均）
    """
    result = np.zeros(3)
    for color_name, ratio in ratios.items():
        if ratio > 0:
            result += BASE_COLORS[color_name] * ratio
    return result


def blend_colors_array(ratios: np.ndarray, color_matrix: np.ndarray) -> np.ndarray:
    """配列ベースの混色計算（RGB平均）"""
    return np.dot(ratios, color_matrix)


def calculate_color_difference(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    """2色間のユークリッド距離（色差）を計算"""
    return np.sqrt(np.sum((rgb1 - rgb2) ** 2))


def find_closest_colors(target_rgb: np.ndarray, n: int = 6) -> list:
    """
    目標色に最も近い色をLab色空間で検索し、上位n色を返す

    Args:
        target_rgb: 目標のRGB値
        n: 返す色の数

    Returns:
        [(色名, Lab距離), ...] のリスト（距離順）
    """
    target_lab = rgb_to_lab(target_rgb)
    distances = []

    for name, rgb in BASE_COLORS.items():
        color_lab = rgb_to_lab(rgb)
        dist = delta_e_cie76(target_lab, color_lab)
        distances.append((name, dist))

    # 距離でソート
    distances.sort(key=lambda x: x[1])
    return distances[:n]


def calculate_mix(target_rgb: np.ndarray, use_all_colors: bool = True, use_lab: bool = True, max_colors: int = 6) -> dict:
    """
    ターゲット色に最も近い配合比率を計算（ベース色基準アプローチ）

    戦略:
    1. 目標色に最も近い色をベース色として選択
    2. ベース色に近い候補色を選定（Lab距離順）
    3. ベース色を高比率で維持しながら最適化

    Args:
        target_rgb: 目標のRGB値 (numpy array)
        use_all_colors: 全色を使用するかどうか
        use_lab: Lab色空間で最適化するかどうか
        max_colors: 使用する最大色数（デフォルト6）

    Returns:
        配合比率の辞書
    """
    target_lab = rgb_to_lab(target_rgb)

    # ステップ1: 目標色に最も近い色を見つける
    closest_colors = find_closest_colors(target_rgb, n=max_colors * 2)
    base_color_name = closest_colors[0][0]
    base_color_dist = closest_colors[0][1]

    # ベース色だけで十分近い場合（ΔE < 3は人間の目では区別困難）
    if base_color_dist < 3.0:
        return {base_color_name: 1.0}

    # ステップ2: 候補色を選定
    # ベース色 + 調整に有効な色を選ぶ
    candidate_names = [base_color_name]

    # 目標色との差分を計算
    base_rgb = BASE_COLORS[base_color_name]
    diff_rgb = target_rgb - base_rgb
    diff_lab = target_lab - rgb_to_lab(base_rgb)

    # 差分を補正できる色を優先的に選ぶ
    color_scores = []
    for name, dist in closest_colors[1:]:
        if name == base_color_name:
            continue
        color_rgb = BASE_COLORS[name]
        color_lab = rgb_to_lab(color_rgb)

        # この色がベース色からどの方向に変化させるか
        direction_rgb = color_rgb - base_rgb
        direction_lab = color_lab - rgb_to_lab(base_rgb)

        # 差分と同じ方向に動かせる色を評価（内積）
        alignment_lab = np.dot(diff_lab, direction_lab)
        alignment_rgb = np.dot(diff_rgb, direction_rgb)

        # 距離も考慮（近い色の方が使いやすい）
        score = alignment_lab / (dist + 1)
        color_scores.append((name, score, dist))

    # スコア順にソート
    color_scores.sort(key=lambda x: -x[1])

    # 候補色を追加（最大max_colors - 1色）
    for name, score, dist in color_scores:
        if len(candidate_names) >= max_colors:
            break
        # 有効な方向に動かせる色、または近い色を採用
        if score > 0 or dist < 30:
            candidate_names.append(name)

    # 候補が少ない場合、近い色を追加
    if len(candidate_names) < max_colors:
        for name, dist in closest_colors:
            if name not in candidate_names:
                candidate_names.append(name)
            if len(candidate_names) >= max_colors:
                break

    # ステップ3: 選ばれた色で最適化
    n_candidates = len(candidate_names)
    candidate_matrix = np.array([BASE_COLORS[name] for name in candidate_names])

    def objective(ratios):
        """目的関数: Lab色空間での色差（ΔE）の最小化（Kubelka-Munk混色）"""
        blended_rgb = blend_colors_array(ratios, candidate_matrix)
        blended_rgb = np.clip(blended_rgb, 0, 255)
        blended_lab = rgb_to_lab(blended_rgb)
        return delta_e_cie76(blended_lab, target_lab)

    # 制約条件: 合計 = 1
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    # 境界条件: 各比率は0〜1
    bounds = [(0, 1) for _ in range(n_candidates)]

    # 複数の初期値から最適化
    best_result = None
    best_error = float('inf')

    initial_points = []

    # 戦略1: ベース色を主体とした初期値（70%, 50%, 30%）
    for base_ratio in [0.7, 0.5, 0.3]:
        x0 = np.ones(n_candidates) * (1 - base_ratio) / (n_candidates - 1)
        x0[0] = base_ratio  # ベース色のインデックスは0
        initial_points.append(x0)

    # 戦略2: 上位2色を主体
    if n_candidates >= 2:
        x0 = np.zeros(n_candidates)
        x0[0] = 0.6
        x0[1] = 0.4
        initial_points.append(x0)

        x0 = np.zeros(n_candidates)
        x0[0] = 0.5
        x0[1] = 0.3
        if n_candidates >= 3:
            x0[2] = 0.2
        else:
            x0[1] = 0.5
        initial_points.append(x0)

    # 戦略3: 均等配分
    initial_points.append(np.ones(n_candidates) / n_candidates)

    # 戦略4: ランダム（ベース色重視）
    np.random.seed(42)
    for _ in range(5):
        x0 = np.random.rand(n_candidates)
        x0[0] *= 2  # ベース色を重視
        x0 = x0 / x0.sum()
        initial_points.append(x0)

    for x0 in initial_points:
        try:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 2000},
            )
            if result.fun < best_error:
                best_error = result.fun
                best_result = result
        except:
            continue

    if best_result is None:
        return {base_color_name: 1.0}

    # 結果を辞書に変換（1%未満は0に）
    ratios = {}
    for i, name in enumerate(candidate_names):
        ratio = best_result.x[i]
        if ratio >= 0.01:  # 1%以上のみ
            ratios[name] = round(ratio, 4)

    # 正規化（合計を1に）
    total = sum(ratios.values())
    if total > 0:
        ratios = {k: round(v / total, 4) for k, v in ratios.items()}

    return ratios


def calculate_mix_result(target_rgb: np.ndarray, delta_e_threshold: float = 5.0) -> dict:
    """
    配合計算の完全な結果を返す（適応的色数調整）

    ΔE >= threshold の場合、色数を段階的に増やして再最適化する。

    Args:
        target_rgb: 目標のRGB値
        delta_e_threshold: この値以上の場合、色数を増やす（デフォルト5.0）

    Returns:
        {
            "target": {"rgb": [r,g,b], "hex": "#RRGGBB"},
            "base_color": ベース色の情報,
            "ratios": {"001-Black": 0.2, ...},
            "result": {"rgb": [r,g,b], "hex": "#RRGGBB"},
            "error": 誤差率(%)
            "delta_e": Lab色空間での色差
            "colors_used": 使用色数
        }
    """
    # ベース色を特定
    closest = find_closest_colors(target_rgb, n=1)[0]
    base_color_name = closest[0]
    base_color_dist = closest[1]

    target_lab = rgb_to_lab(target_rgb)
    total_colors = len(BASE_COLORS)

    # 段階的に色数を増やして最適化
    best_ratios = None
    best_delta_e = float('inf')
    best_result_rgb = None

    # 6色から開始、2色ずつ増やす
    for max_colors in range(6, total_colors + 1, 2):
        ratios = calculate_mix(target_rgb, use_lab=True, max_colors=max_colors)
        result_rgb = blend_colors(ratios)
        result_rgb = np.clip(result_rgb, 0, 255)

        result_lab = rgb_to_lab(result_rgb)
        delta_e = delta_e_cie76(target_lab, result_lab)

        # より良い結果なら更新
        if delta_e < best_delta_e:
            best_delta_e = delta_e
            best_ratios = ratios
            best_result_rgb = result_rgb

        # ΔE < threshold なら終了
        if delta_e < delta_e_threshold:
            break

    # 最後に全色で試す（まだしきい値を超えている場合）
    if best_delta_e >= delta_e_threshold:
        ratios = calculate_mix(target_rgb, use_lab=True, max_colors=total_colors)
        result_rgb = blend_colors(ratios)
        result_rgb = np.clip(result_rgb, 0, 255)

        result_lab = rgb_to_lab(result_rgb)
        delta_e = delta_e_cie76(target_lab, result_lab)

        if delta_e < best_delta_e:
            best_delta_e = delta_e
            best_ratios = ratios
            best_result_rgb = result_rgb

    # RGB誤差
    rgb_error = calculate_color_difference(target_rgb, best_result_rgb)
    rgb_error_percent = (rgb_error / 441.67) * 100  # 最大誤差(白→黒)で正規化

    # ΔEを%に変換（ΔE=100が最大と仮定）
    delta_e_percent = min(best_delta_e, 100)

    return {
        "target": {"rgb": target_rgb.tolist(), "hex": rgb_to_hex(target_rgb)},
        "base_color": {
            "name": base_color_name,
            "name_jp": COLOR_NAMES_JP.get(base_color_name, base_color_name),
            "rgb": BASE_COLORS[base_color_name].tolist(),
            "hex": rgb_to_hex(BASE_COLORS[base_color_name]),
            "delta_e": round(base_color_dist, 2),
        },
        "ratios": best_ratios,
        "ratios_percent": {k: round(v * 100, 1) for k, v in best_ratios.items()},
        "result": {"rgb": best_result_rgb.tolist(), "hex": rgb_to_hex(best_result_rgb)},
        "error": round(delta_e_percent, 2),  # Lab色差を使用
        "delta_e": round(best_delta_e, 2),
        "rgb_error": round(rgb_error_percent, 2),
        "colors_used": len(best_ratios),
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

    # ベース色の情報
    base = result.get("base_color", {})
    if base:
        print(f"\nベース色: {base['name']} ({base['name_jp']})")
        print(f"  {base['hex']} - 目標との差: ΔE={base['delta_e']}")

    colors_used = result.get("colors_used", len(result["ratios_percent"]))
    print(f"\n--- 配合比率 ({colors_used}色使用) ---")

    for color_name, percent in sorted(
        result["ratios_percent"].items(), key=lambda x: -x[1]
    ):
        jp_name = COLOR_NAMES_JP.get(color_name, color_name)
        bar = "█" * int(percent / 5) + "░" * (20 - int(percent / 5))
        print(f"{color_name} ({jp_name}): {bar} {percent}%")

    print(f"\n再現色: {result['result']['hex']}")
    print(f"RGB: [" + ", ".join(f"{int(v)}" for v in result["result"]["rgb"]) + "]")
    print(f"色差: ΔE={result['delta_e']} (誤差: {result['error']}%)")
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
