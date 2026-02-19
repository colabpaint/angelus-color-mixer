#!/usr/bin/env python3
"""
Angelus Paint Color Mixer - Flask API Server
=============================================

使用方法:
    python app.py

ブラウザで http://localhost:5000 を開く
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# color_mixerをインポート
from color_mixer import calculate_mix_result, hex_to_rgb

app = Flask(__name__, static_folder=".")
CORS(app)


@app.route("/")
def index():
    """メインページ"""
    return send_from_directory(".", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """静的ファイル配信"""
    return send_from_directory(".", filename)


@app.route("/api/mix", methods=["POST"])
def mix_colors():
    """
    配合比率計算API

    Request JSON:
        {"hex": "#FF8800"} または {"r": 255, "g": 136, "b": 0}

    Response JSON:
        {
            "target": {"rgb": [255, 136, 0], "hex": "#FF8800"},
            "ratios": {"001-Black": 0.0, "005-White": 0.15, ...},
            "ratios_percent": {"005-White": 15.0, ...},
            "result": {"rgb": [252, 140, 5], "hex": "#FC8C05"},
            "error": 1.2
        }
    """
    try:
        data = request.get_json()

        if "hex" in data:
            target_rgb = hex_to_rgb(data["hex"])
        elif all(k in data for k in ["r", "g", "b"]):
            import numpy as np

            target_rgb = np.array([data["r"], data["g"], data["b"]])
        else:
            return jsonify({"error": "Invalid input. Provide 'hex' or 'r', 'g', 'b'"}), 400

        result = calculate_mix_result(target_rgb)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/colors", methods=["GET"])
def get_colors():
    """利用可能な基本色を返す"""
    from color_mixer import BASE_COLORS, COLOR_NAMES_JP, CHART_COLORS

    colors = []
    for code, rgb in BASE_COLORS.items():
        colors.append(
            {
                "code": code,
                "name_jp": COLOR_NAMES_JP[code],
                "rgb": rgb.tolist(),
                "chart_color": CHART_COLORS[code],
            }
        )
    return jsonify(colors)


if __name__ == "__main__":
    print("=" * 50)
    print("Angelus Paint Color Mixer")
    print("=" * 50)
    print("\nサーバーを起動しています...")
    print("ブラウザで http://localhost:5000 を開いてください")
    print("\n終了するには Ctrl+C を押してください")
    print("=" * 50)

    app.run(debug=True, host="0.0.0.0", port=5000)
