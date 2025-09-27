title = Polar Align Assistant
package.name = polaralign
package.domain = org.astronomy

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json

version = 1.0
version.regex = __version__ = ['"](.*)['"]
version.filename = %(source.dir)s/main.py

requirements = python3,kivy,opencv-python,opencv-python-headless,numpy

android.permissions = CAMERA,INTERNET,WRITE_EXTERNAL_STORAGE

android.api = 30
android.minapi = 21
android.sdk = 30
android.ndk = 23b
android.ndk_api = 21

p4a.branch = master
android.arch = arm64-v8a

[buildozer]
log_level = 2

warn_on_root = 1
