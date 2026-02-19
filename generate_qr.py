#!/usr/bin/env python3
"""QRコード生成スクリプト"""
import qrcode

url = "https://colabpaint.github.io/angelus-color-mixer/"

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(url)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.save("QRコード.png")
print("QRコード.png を保存しました")
