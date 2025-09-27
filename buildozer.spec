[app]
title = Polar Align Assistant
package.name = polaralign
package.domain = org.astronomy

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json

version = 1.2
version.regex = __version__ = ['"](.*)['"]
version.filename = %(source.dir)s/main.py

requirements = python3,kivy==2.1.0,opencv-python==4.5.5.64,opencv-python-headless==4.5.5.64,numpy==1.21.6,android

android.permissions = CAMERA,INTERNET,WRITE_EXTERNAL_STORAGE,ACCESS_FINE_LOCATION

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

[app_config]
orientation = portrait
fullscreen = 0
