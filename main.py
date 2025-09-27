from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.storage.jsonstore import JsonStore

import cv2
import numpy as np
import math
import datetime
import json
import os

class ConfigManager:
    def __init__(self):
        self.store = JsonStore('polar_align_config.json')
        self.default_config = {
            "gps": {
                "latitude": 50.51783,
                "longitude": 2.86613,
                "altitude": 25
            },
            "camera": {
                "width": 1280,
                "height": 720,
                "fps": 30
            },
            "calibration": {
                "min_radius": 25,
                "max_radius": 150,
                "detection_interval": 5
            }
        }
        self.load_config()

    def load_config(self):
        try:
            if self.store.exists('config'):
                self.config = self.store.get('config')
            else:
                self.config = self.default_config
                self.save_config()
        except Exception as e:
            Logger.error(f"Config load error: {e}")
            self.config = self.default_config

    def save_config(self):
        try:
            self.store.put('config', **self.config)
        except Exception as e:
            Logger.error(f"Config save error: {e}")

    def get(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)

    def set(self, section, key, value):
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()

class PolarAlignCalculator:
    def __init__(self, config_manager):
        self.config = config_manager
        self.reticle_center = None
        self.reticle_radius = None
        self.reticle_locked = False
        self.frame_count = 0
        self.detection_interval = self.config.get("calibration", "detection_interval", 5)

    def detect_reticle(self, frame):
        """Détection améliorée du réticule avec tracking"""
        # Si réticule verrouillé, ne pas redétecter
        if self.reticle_locked and self.reticle_center and self.reticle_radius:
            return True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Essayer le tracking si réticule connu
        if self.reticle_center and self.reticle_radius:
            x, y = self.reticle_center
            r = self.reticle_radius

            # Définir ROI autour du réticule précédent
            y1 = max(0, y - r - 30)
            y2 = min(frame.shape[0], y + r + 30)
            x1 = max(0, x - r - 30)
            x2 = min(frame.shape[1], x + r + 30)
            roi = gray[y1:y2, x1:x2]

            # Détecter cercles dans ROI
            circles = cv2.HoughCircles(
                roi,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=40,
                param2=25,
                minRadius=max(15, r - 15),
                maxRadius=r + 15
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # Ajuster coordonnées à l'image complète
                circles[:, 0] += x1
                circles[:, 1] += y1
                # Choisir le cercle le plus proche
                closest = min(circles, key=lambda c: math.sqrt((c[0] - x) ** 2 + (c[1] - y) ** 2))
                self.reticle_center = (closest[0], closest[1])
                self.reticle_radius = closest[2]
                return True

        # Détection complète si pas de réticule ou tracking échoué
        min_radius = self.config.get("calibration", "min_radius", 25)
        max_radius = self.config.get("calibration", "max_radius", 150)

        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min(frame.shape[0], frame.shape[1]) // 2,
            param1=40,
            param2=25,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            closest = min(circles, key=lambda c: math.sqrt((c[0] - center_x) ** 2 + (c[1] - center_y) ** 2))
            self.reticle_center = (closest[0], closest[1])
            self.reticle_radius = closest[2]
            Logger.info(f"Réticule détecté: center={self.reticle_center}, radius={self.reticle_radius}")
            return True

        return False

    def calculate_polaris_position(self):
        """Calcul réaliste de la position de Polaris"""
        if not self.reticle_center or not self.reticle_radius:
            return None

        try:
            # Temps UTC actuel
            utc_time = datetime.datetime.now(datetime.timezone.utc)

            # Coordonnées fixes de Polaris (approximatives mais réalistes)
            polaris_ra_hours = 2.53030102  # Ascension droite en heures
            polaris_dec_deg = 89.26413805  # Déclinaison en degrés

            # Calcul de l'angle horaire (simplifié mais réaliste)
            current_time = utc_time.hour + utc_time.minute/60.0 + utc_time.second/3600.0
            lst = (current_time + self.config.get("gps", "longitude", 0) / 15.0) % 24  # Temps sidéral local
            ha_hours = (lst - polaris_ra_hours) % 24
            ha_deg = ha_hours * 15  # Convertir en degrés

            # Distance du pôle céleste (90° - déclinaison de Polaris)
            distance_from_pole = 90.0 - polaris_dec_deg  # Environ 0.74°

            # Convertir en coordonnées du réticule
            center_x, center_y = self.reticle_center
            radius_norm = min(distance_from_pole / 45.0, 1.0)  # Normaliser sur 45°
            radius = self.reticle_radius * radius_norm

            # Orientation astronomique correcte
            # 0h HA = Sud (bas), 6h = Ouest (droite), 12h = Nord (haut), 宏大 = Est (gauche)
            target_x = int(center_x + radius * math.sin(math.radians(ha_deg)))
            target_y = int(center_y - radius * math.cos(math.radians(ha_deg)))  # Négatif pour orientation correcte

            return (target_x, target_y, ha_deg, distance_from_pole)
        except Exception as e:
            Logger.error(f"Erreur calcul Polaris: {e}")
            return None

    def calculate_direction(self, target_pos):
        """Calcul de la direction avec offset"""
        if not self.reticle_center or not target_pos:
            return "INCONNU", (0, 0)

        dx = target_pos[0] - self.reticle_center[0]
        dy = target_pos[1] - self.reticle_center[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 12:
            return "CENTRE", (0, 0)

        if abs(dx) > abs(dy):
            direction = "OUEST" if dx > 0 else "EST"
        else:
            direction = "SUD" if dy > 0 else "NORD"

        return direction, (dx, dy)

    def draw_overlay(self, frame):
        """Dessin amélioré de l'overlay"""
        if self.reticle_center and self.reticle_radius:
            # Réticule (vert)
            cv2.circle(frame, self.reticle_center, self.reticle_radius, (0, 255, 0), 1)
            cv2.line(frame,
                     (self.reticle_center[0] - 12, self.reticle_center[1]),
                     (self.reticle_center[0] + 12, self.reticle_center[1]),
                     (0, 255, 0), 1)
            cv2.line(frame,
                     (self.reticle_center[0], self.reticle_center[1] - 12),
                     (self.reticle_center[0], self.reticle_center[1] + 12),
                     (0, 255, 0), 1)

            # Marquages horaires (0h/12h, 3h, 6h, 9h)
            hours_labels = [(0, "0h/12h"), (3, "3h"), (6, "6h"), (9, "9h")]
            for hour, label in hours_labels:
                angle_deg = (hour * 30) - 90  # -90 pour aligner 0h avec le bas
                angle_rad = math.radians(angle_deg)
                x1 = int(self.reticle_center[0] + (self.reticle_radius - 12) * math.cos(angle_rad))
                y1 = int(self.reticle_center[1] + (self.reticle_radius - 12) * math.sin(angle_rad))
                x2 = int(self.reticle_center[0] + self.reticle_radius * math.cos(angle_rad))
                y2 = int(self.reticle_center[1] + self.reticle_radius * math.sin(angle_rad))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                label_x = int(self.reticle_center[0] + (self.reticle_radius + 15) * math.cos(angle_rad))
                label_y = int(self.reticle_center[1] + (self.reticle_radius + 15) * math.sin(angle_rad)) + 5
                cv2.putText(frame, label, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

        # Position de Polaris
        polaris_data = self.calculate_polaris_position()
        if polaris_data:
            target_x, target_y, ha_deg, distance_from_pole = polaris_data
            target_pos = (target_x, target_y)

            # Marqueur Polaris (rouge)
            cv2.circle(frame, target_pos, 4, (0, 0, 255), -1)
            cv2.circle(frame, target_pos, 6, (0, 0, 255), 1)
            cv2.putText(frame, "POLARIS",
                        (target_pos[0] + 12, target_pos[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Direction et informations
            direction, offset = self.calculate_direction(target_pos)

            # Textes d'information
            cv2.putText(frame, f"Direction: {direction}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"HA: {ha_deg:.1f}°", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Distance: {distance_from_pole:.1f}°", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = ConfigManager()
        self.calculator = PolarAlignCalculator(self.config)

        # Layout principal
        main_layout = BoxLayout(orientation='vertical')

        # Titre
        main_layout.add_widget(Label(text='🧭 Polar Alignment Assistant',
                                   size_hint_y=0.1,
                                   font_size='20sp'))

        # Zone caméra
        self.camera_image = Image()
        main_layout.add_widget(self.camera_image)

        # Boutons de contrôle
        controls = BoxLayout(size_hint_y=0.2, spacing=10, padding=10)

        self.start_btn = Button(text='▶ Démarrer Caméra')
        self.start_btn.bind(on_press=self.toggle_camera)

        recalibrate_btn = Button(text='🔄 Recalibrer')
        recalibrate_btn.bind(on_press=self.recalibrate)

        settings_btn = Button(text='⚙️ Paramètres')
        settings_btn.bind(on_press=self.go_to_settings)

        controls.add_widget(self.start_btn)
        controls.add_widget(recalibrate_btn)
        controls.add_widget(settings_btn)

        main_layout.add_widget(controls)
        self.add_widget(main_layout)

        self.capture = None
        self.camera_running = False

    def toggle_camera(self, instance):
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        try:
            self.capture = cv2.VideoCapture(0)
            if self.capture.isOpened():
                # Configurer la caméra
                width = self.config.get("camera", "width", 1280)
                height = self.config.get("camera", "height", 720)
                fps = self.config.get("camera", "fps", 30)

                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.capture.set(cv2.CAP_PROP_FPS, fps)

                self.camera_running = True
                self.start_btn.text = '⏹ Arrêter Caméra'
                Clock.schedule_interval(self.update_frame, 1.0/30.0)
            else:
                self.show_popup("Erreur", "Impossible d'ouvrir la caméra")
        except Exception as e:
            self.show_popup("Erreur", f"Erreur caméra: {str(e)}")

    def stop_camera(self):
        if self.camera_running:
            Clock.unschedule(self.update_frame)
            if self.capture:
                self.capture.release()
            self.camera_running = False
            self.start_btn.text = '▶ Démarrer Caméra'

    def update_frame(self, dt):
        if self.capture and self.camera_running:
            ret, frame = self.capture.read()
            if ret:
                self.calculator.frame_count += 1

                # Redétecter périodiquement si pas verrouillé
                if not self.calculator.reticle_locked and self.calculator.frame_count % self.calculator.detection_interval == 0:
                    self.calculator.reticle_center = None
                    self.calculator.reticle_radius = None

                # Détecter réticule si nécessaire
                if not self.calculator.reticle_center:
                    if self.calculator.detect_reticle(frame):
                        self.calculator.reticle_locked = True

                # Dessiner l'overlay
                self.calculator.draw_overlay(frame)

                # Convertir pour Kivy
                buf = cv2.flip(frame, 0).tostring()
                texture = Texture.create(
                    size=(frame.shape[1], frame.shape[0]),
                    colorfmt='bgr'
                )
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.camera_image.texture = texture

    def recalibrate(self, instance):
        self.calculator.reticle_locked = False
        self.calculator.reticle_center = None
        self.calculator.reticle_radius = None
        self.show_popup("Info", "Recalibrage lancé. Placez le réticule au centre.")

    def go_to_settings(self, instance):
        self.manager.current = 'settings'

    def show_popup(self, title, message):
        popup = Popup(title=title, content=Label(text=message),
                     size_hint=(0.8, 0.4))
        popup.open()

    def on_leave(self):
        self.stop_camera()

class SettingsScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = ConfigManager()

        # Layout principal avec onglets
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Titre
        main_layout.add_widget(Label(text='⚙️ Paramètres', size_hint_y=0.1, font_size='20sp'))

        # Onglets
        tab_panel = TabbedPanel(do_default_tab=False)

        # Onglet GPS
        gps_tab = TabbedPanelItem(text='📍 GPS')
        gps_layout = GridLayout(cols=2, spacing=10, padding=10)

        gps_layout.add_widget(Label(text='Latitude:'))
        self.lat_input = TextInput(text=str(self.config.get("gps", "latitude", "50.51783")))
        gps_layout.add_widget(self.lat_input)

        gps_layout.add_widget(Label(text='Longitude:'))
        self.lon_input = TextInput(text=str(self.config.get("gps", "longitude", "2.86613")))
        gps_layout.add_widget(self.lon_input)

        gps_layout.add_widget(Label(text='Altitude (m):'))
        self.alt_input = TextInput(text=str(self.config.get("gps", "altitude", "25")))
        gps_layout.add_widget(self.alt_input)

        gps_tab.add_widget(gps_layout)
        tab_panel.add_widget(gps_tab)

        # Onglet Caméra
        cam_tab = TabbedPanelItem(text='📷 Caméra')
        cam_layout = GridLayout(cols=2, spacing=10, padding=10)

        cam_layout.add_widget(Label(text='Largeur:'))
        self.width_input = TextInput(text=str(self.config.get("camera", "width", "1280")))
        cam_layout.add_widget(self.width_input)

        cam_layout.add_widget(Label(text='Hauteur:'))
        self.height_input = TextInput(text=str(self.config.get("camera", "height", "720")))
        cam_layout.add_widget(self.height_input)

        cam_layout.add_widget(Label(text='FPS:'))
        self.fps_input = TextInput(text=str(self.config.get("camera", "fps", "30")))
        cam_layout.add_widget(self.fps_input)

        cam_tab.add_widget(cam_layout)
        tab_panel.add_widget(cam_tab)

        # Onglet Calibration
        cal_tab = TabbedPanelItem(text='🔧 Calibration')
        cal_layout = GridLayout(cols=2, spacing=10, padding=10)

        cal_layout.add_widget(Label(text='Rayon Min:'))
        self.min_radius_input = TextInput(text=str(self.config.get("calibration", "min_radius", "25")))
        cal_layout.add_widget(self.min_radius_input)

        cal_layout.add_widget(Label(text='Rayon Max:'))
        self.max_radius_input = TextInput(text=str(self.config.get("calibration", "max_radius", "150")))
        cal_layout.add_widget(self.max_radius_input)

        cal_layout.add_widget(Label(text='Intervalle (frames):'))
        self.interval_input = TextInput(text=str(self.config.get("calibration", "detection_interval", "5")))
        cal_layout.add_widget(self.interval_input)

        cal_tab.add_widget(cal_layout)
        tab_panel.add_widget(cal_tab)

        main_layout.add_widget(tab_panel)

        # Boutons
        btn_layout = BoxLayout(size_hint_y=0.2, spacing=10, padding=10)

        save_btn = Button(text='💾 Sauvegarder')
        save_btn.bind(on_press=self.save_settings)

        back_btn = Button(text='⬅️ Retour')
        back_btn.bind(on_press=self.go_back)

        btn_layout.add_widget(save_btn)
        btn_layout.add_widget(back_btn)

        main_layout.add_widget(btn_layout)
        self.add_widget(main_layout)

    def save_settings(self, instance):
        try:
            # GPS
            self.config.set("gps", "latitude", float(self.lat_input.text))
            self.config.set("gps", "longitude", float(self.lon_input.text))
            self.config.set("gps", "altitude", float(self.alt_input.text))

            # Camera
            self.config.set("camera", "width", int(self.width_input.text))
            self.config.set("camera", "height", int(self.height_input.text))
            self.config.set("camera", "fps", int(self.fps_input.text))

            # Calibration
            self.config.set("calibration", "min_radius", int(self.min_radius_input.text))
            self.config.set("calibration", "max_radius", int(self.max_radius_input.text))
            self.config.set("calibration", "detection_interval", int(self.interval_input.text))

            self.show_popup("Succès", "Paramètres sauvegardés avec succès!")
        except Exception as e:
            self.show_popup("Erreur", f"Erreur de sauvegarde: {str(e)}")

    def go_back(self, instance):
        self.manager.current = 'camera'

    def show_popup(self, title, message):
        popup = Popup(title=title, content=Label(text=message),
                     size_hint=(0.8, 0.4))
        popup.open()

class PolarAlignApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(CameraScreen(name='camera'))
        sm.add_widget(SettingsScreen(name='settings'))
        return sm

if __name__ == '__main__':
    PolarAlignApp().run()
