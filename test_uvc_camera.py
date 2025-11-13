#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Test pour Caméras USB UVC
====================================

Ce script détecte et teste les caméras USB UVC disponibles.
Il affiche les propriétés de chaque caméra détectée.

Usage:
    python test_uvc_camera.py
"""

import cv2
import sys

def test_camera(index):
    """Teste une caméra à l'index donné"""
    print(f"\n{'='*60}")
    print(f"Test de la caméra {index}...")
    print(f"{'='*60}")
    
    # Essayer différents backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow (Windows)"),
        (cv2.CAP_MSMF, "Media Foundation (Windows)"),
        (cv2.CAP_ANY, "Backend automatique")
    ]
    
    for backend, backend_name in backends:
        try:
            print(f"\n📹 Tentative avec backend: {backend_name}")
            cap = cv2.VideoCapture(index, backend)
            
            if cap.isOpened():
                # Récupérer les propriétés
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                print(f"   ✅ CAMÉRA DÉTECTÉE !")
                print(f"   Backend: {backend_name}")
                print(f"   Résolution: {width}x{height}")
                print(f"   FPS: {fps}")
                
                # Test capture d'image
                print(f"\n   Test de capture d'image...")
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    print(f"   ✅ Capture OK - Image {frame.shape[1]}x{frame.shape[0]}")
                    
                    # Tester plusieurs résolutions courantes
                    print(f"\n   Test des résolutions supportées:")
                    resolutions = [
                        (640, 480, "VGA"),
                        (800, 600, "SVGA"),
                        (1280, 720, "HD 720p"),
                        (1920, 1080, "Full HD")
                    ]
                    
                    for w, h, name in resolutions:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        if actual_w == w and actual_h == h:
                            print(f"   ✅ {name} ({w}x{h}) - Supportée")
                        else:
                            print(f"   ⚠️  {name} ({w}x{h}) - Non supportée (obtenu {actual_w}x{actual_h})")
                    
                    cap.release()
                    
                    # Recommandation
                    print(f"\n{'='*60}")
                    print(f"✅ CONFIGURATION RECOMMANDÉE POUR config.ini:")
                    print(f"{'='*60}")
                    print(f"[CAMERA]")
                    print(f"index = {index}")
                    print(f"width = 640")
                    print(f"height = 480")
                    print(f"fps = 10")
                    print(f"{'='*60}\n")
                    
                    return True
                else:
                    print(f"   ❌ Erreur de capture d'image")
                    cap.release()
            else:
                print(f"   ❌ Impossible d'ouvrir la caméra avec ce backend")
        
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    return False

def main():
    """Fonction principale"""
    print("\n" + "="*60)
    print("🎥 TEST DE CAMÉRAS USB UVC")
    print("="*60)
    print("\nCe script va tester les 5 premiers index de caméra.")
    print("Veuillez patienter...\n")
    
    cameras_found = []
    
    # Tester les 5 premiers index
    for i in range(5):
        if test_camera(i):
            cameras_found.append(i)
        else:
            print(f"\n❌ Aucune caméra détectée à l'index {i}")
    
    # Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ")
    print("="*60)
    
    if cameras_found:
        print(f"\n✅ {len(cameras_found)} caméra(s) détectée(s) :")
        for idx in cameras_found:
            print(f"   • Caméra à l'index {idx}")
        
        print(f"\n💡 CONSEIL:")
        print(f"   Utilisez index = {cameras_found[0]} dans config.ini")
        print(f"   (ou testez les autres si vous avez plusieurs caméras)")
        
        print(f"\n📝 PROCHAINES ÉTAPES:")
        print(f"   1. Éditez config.ini")
        print(f"   2. Changez 'index = {cameras_found[0]}'")
        print(f"   3. Lancez python polar_align.py")
        print(f"   4. Ouvrez http://localhost:5000")
    else:
        print(f"\n❌ Aucune caméra USB UVC détectée !")
        print(f"\n🔧 VÉRIFICATIONS:")
        print(f"   1. La caméra est-elle branchée ?")
        print(f"   2. Vérifiez dans Gestionnaire de périphériques (Windows)")
        print(f"   3. Fermez les autres applications utilisant la caméra")
        print(f"   4. Essayez un autre port USB")
        print(f"   5. Redémarrez l'ordinateur")
    
    print("\n" + "="*60)
    print("✅ Test terminé !")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Erreur inattendue: {e}")
        sys.exit(1)

