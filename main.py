from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.core.camera import Camera

import cv2
import numpy as np
import math
import json
import os

class PolarAlignApp(App):
    def build(self):
        self.title = "Polar Align Assistant"

        # Layout principal
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Titre
        layout.add_widget(Label(text='🧭 Polar Align Assistant',
                              size_hint_y=0.1,
                              font_size='20sp'))

        # Zone d'affichage caméra
        self.camera_image = Image()
        layout.add_widget(self.camera_image)

        # Zone d'information
        self.info_label = Label(text='Appuyez sur Démarrer pour commencer',
                               size_hint_y=0.2,
                               halign='center')
        layout.add_widget(self.info_label)

        # Boutons de contrôle
        button_layout = BoxLayout(size_hint_y=0.2, spacing=10)

        self.start_btn = Button(text='Démarrer Caméra')
        self.start_btn.bind(on_press=self.toggle_camera)

        self.detect_btn = Button(text='Détecter Réticule')
        self.detect_btn.bind(on_press=self.detect_reticle)

        self.recalibrate_btn = Button(text='Recalibrer')
        self.recalibrate_btn.bind(on_press=self.recalibrate)

        button_layout.add_widget(self.start_btn)
        button_layout.add_widget(self.detect_btn)
        button_layout.add_widget(self.recalibrate_btn)

        layout.add_widget(button_layout)

        # Variables
        self.capture = None
        self.camera_running = False
        self.reticle_center = None
        self.reticle_radius = None

        return layout

    def toggle_camera(self, instance):
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        try:
            self.capture = cv2.VideoCapture(0)
            if self.capture.isOpened():
                self.camera_running = True
                self.start_btn.text = 'Arrêter Caméra'
                Clock.schedule_interval(self.update_frame, 1.0/30.0)
                self.info_label.text = 'Caméra démarrée. Pointez vers le réticule.'
            else:
                self.info_label.text = '❌ Impossible d\'ouvrir la caméra'
        except Exception as e:
            self.info_label.text = f'❌ Erreur: {str(e)}'

    def stop_camera(self):
        if self.camera_running:
            Clock.unschedule(self.update_frame)
            if self.capture:
                self.capture.release()
            self.camera_running = False
            self.start_btn.text = 'Démarrer Caméra'
            self.info_label.text = 'Caméra arrêtée'

    def update_frame(self, dt):
        if self.capture and self.camera_running:
            ret, frame = self.capture.read()
            if ret:
                # Dessiner le réticule si détecté
                if self.reticle_center and self.reticle_radius:
                    cv2.circle(frame, self.reticle_center, self.reticle_radius, (0, 255, 0), 2)
                    cv2.line(frame,
                            (self.reticle_center[0] - 20, self.reticle_center[1]),
                            (self.reticle_center[0] + 20, self.reticle_center[1]),
                            (0, 255, 0), 2)
                    cv2.line(frame,
                            (self.reticle_center[0], self.reticle_center[1] - 20),
                            (self.reticle_center[0], self.reticle_center[1] + 20),
                            (0, 255, 0), 2)

                # Convertir pour Kivy
                buf = cv2.flip(frame, 0).tostring()
                texture = Texture.create(
                    size=(frame.shape[1], frame.shape[0]),
                    colorfmt='bgr'
                )
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.camera_image.texture = texture

    def detect_reticle(self, instance):
        if not self.camera_running:
            self.info_label.text = '❌ Démarrer la caméra d\'abord'
            return

        # Capture d'une frame pour détection
        ret, frame = self.capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=min(frame.shape[0], frame.shape[1]) // 2,
                param1=50,
                param2=30,
                minRadius=20,
                maxRadius=200
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                closest = min(circles, key=lambda c: math.sqrt((c[0] - center_x) ** 2 + (c[1] - center_y) ** 2))
                self.reticle_center = (closest[0], closest[1])
                self.reticle_radius = closest[2]
                self.info_label.text = f'✅ Réticule détecté ! Position: {self.reticle_center}'
            else:
                self.info_label.text = '❌ Réticule non détecté. Réessayez.'

    def recalibrate(self, instance):
        self.reticle_center = None
        self.reticle_radius = None
        self.info_label.text = '🔄 Recalibrage. Appuyez sur "Détecter Réticule"'

if __name__ == '__main__':
    PolarAlignApp().run()