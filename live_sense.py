"""
Live Sensory HUD: Real-Time Biometric & Retinal Emotion Sensor
============================================================
Run live with webcam, video file, or test images:
    python live_sense.py
    python live_sense.py --source sample_data/masked_sadness.jpg
    python live_sense.py --camera 0

Hotkeys:
    'Q' or ESC : Quit
    'M'        : Toggle facial landmark wireframe
    'R'        : Toggle 3D retinal gaze vector rays
    'S'        : Save single snapshot frame & JSON telemetry
    'V'        : Toggle Video + Microphone Audio Recording
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import wave
import cv2
import numpy as np

try:
    import sounddevice as sd
    import scipy.io.wavfile as wavfile
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

from engine.detector import SensoryPipeline, SensoryResult
from engine.auth import AuthManager


class AudioRecorder:
    """Records microphone audio in background thread."""

    def __init__(self, sample_rate=44100, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.audio_frames = []
        self._stream = None

    def start(self):
        if not AUDIO_AVAILABLE:
            return
        self.is_recording = True
        self.audio_frames = []

        def callback(indata, frames, time_info, status):
            if self.is_recording:
                self.audio_frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=callback
        )
        self._stream.start()

    def stop(self, out_wav_path: str):
        if not AUDIO_AVAILABLE or not self.is_recording:
            return
        self.is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self.audio_frames:
            all_audio = np.concatenate(self.audio_frames, axis=0)
            # Write 16-bit PCM wav
            scaled = np.int16(all_audio * 32767)
            wavfile.write(out_wav_path, self.sample_rate, scaled)


def run_live(source=0, show_mesh=True, show_rays=True, key=None):
    print("=" * 68)
    print("  HUMAN EMOTION & TRUE SENSE IDENTIFICATION SYSTEM (HUD)")
    print("=" * 68)

    # Security Gating & Session Authorization
    auth_mgr = AuthManager()
    if auth_mgr.auth_enabled:
        token = key or os.getenv("ACCESS_KEY")
        if not token and auth_mgr.admin_master_key:
            token = auth_mgr.admin_master_key

        if not token:
            try:
                token = input("Enter Access Key or Admin Key to unlock: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession aborted.")
                sys.exit(1)

        ok, msg, session = auth_mgr.validate_and_activate(
            token,
            user_info={"name": "CLI User", "phone": "+10000000000", "email": "cli@local.user"},
            require_user_info=False
        )
        if not ok:
            print(f"\n[SECURITY LOCKOUT] Authorization failed: {msg}")
            print("Access is restricted. Run 'python make_key.py' to issue an Access Key.\n")
            sys.exit(1)

        AuthManager.set_current_session(session)
        print(f"[SECURITY] Session authorized ({session.get('key_display', 'AUTH-OK')}).")

    print("Initializing Sensory Pipeline...")

    try:
        pipeline = SensoryPipeline()
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)

    # Check if source is static image file
    if isinstance(source, str) and os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            print(f"Processing static image: {source}")
            img = cv2.imread(source)
            if img is None:
                print(f"Failed to read image: {source}")
                return

            results = pipeline.process_frame_multi(img, is_static=True)
            if results and len(results) > 0:
                annotated = pipeline.draw_hud_multi(img, results, show_mesh=show_mesh, show_gaze_rays=show_rays)
                print(f"\n[MULTI-FACE DETECTED]: {len(results)} face(s) analyzed")
                for r in results:
                    print(f"\n--- Person {r.track_id} ---")
                    print(f"[DIAGNOSIS]: {r.affect.primary_state}")
                    print(f"[SINCERITY]: {r.affect.sincerity_score * 100:.1f}% (Masking: {r.affect.is_masking_detected})")
                    print(f"🧪 Scientific: {r.affect.scientific_justification[:100]}...")
                    print(f"💡 Everyday: {r.affect.layman_justification[:100]}...")

                out_path = "output_analysis.jpg"
                cv2.imwrite(out_path, annotated)
                print(f"\nAnnotated HUD output saved to: {out_path}")
            else:
                print("No face detected in image.")
            pipeline.close()
            return

    # Video or camera capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[WARNING] Could not open camera {source}.")
        print("💡 Note: On macOS, please ensure Terminal has Camera access enabled in:")
        print("   System Settings -> Privacy & Security -> Camera")
        print("Falling back to sample verification mode...")

        samples = [
            "sample_data/masked_sadness.jpg",
            "sample_data/genuine_joy.jpg",
            "sample_data/in_love.jpg",
            "sample_data/neutral_focus.jpg"
        ]
        for sample in samples:
            if os.path.exists(sample):
                img = cv2.imread(sample)
                res = pipeline.process_frame(img)
                if res:
                    out = pipeline.draw_hud(img, res)
                    name = os.path.basename(sample)
                    cv2.imwrite(f"output_{name}", out)
                    print(f"Processed sample {sample} -> output_{name} [{res.affect.primary_state}]")
        pipeline.close()
        return

    # Audio recorder & Video Writer state
    audio_rec = AudioRecorder()
    is_recording_av = False
    video_writer = None
    rec_start_time = 0.0
    temp_vid_path = "temp_rec_video.mp4"
    temp_aud_path = "temp_rec_audio.wav"

    print("Controls:")
    print("  'Q' / ESC : Quit")
    print("  'M'       : Toggle facial landmark wireframe")
    print("  'R'       : Toggle 3D retinal gaze vector rays")
    print("  'S'       : Save single snapshot frame & JSON telemetry")
    print("  'V'       : START / STOP Video + Microphone Audio Recording")

    fps_history = []
    t_prev = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or disconnected.")
                break

            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            fps = 1.0 / dt if dt > 0 else 30.0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            results = pipeline.process_frame_multi(frame, timestamp=t_now)
            if results and len(results) > 0:
                display_frame = pipeline.draw_hud_multi(frame, results, show_mesh=show_mesh, show_gaze_rays=show_rays)
            else:
                display_frame = frame.copy()
                cv2.putText(display_frame, "SEARCHING FOR FACIAL SENSORS...", (30, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 165, 255), 1, cv2.LINE_AA)

            # Draw Recording Indicator
            if is_recording_av:
                rec_duration = int(t_now - rec_start_time)
                mins = rec_duration // 60
                secs = rec_duration % 60
                # Flashing red dot
                if int(t_now * 2) % 2 == 0:
                    cv2.circle(display_frame, (35, 120), 8, (0, 0, 255), -1)
                cv2.putText(display_frame, f"REC (VIDEO + MIC) {mins:02d}:{secs:02d}", (50, 126),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
                if video_writer:
                    video_writer.write(display_frame)

            # FPS counter
            cv2.putText(display_frame, f"{avg_fps:.0f} FPS", (18, display_frame.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

            cv2.imshow("AffectSense AI: Live Retinal & Subconscious Affect HUD", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('m'):
                show_mesh = not show_mesh
            elif key == ord('r'):
                show_rays = not show_rays
            elif key == ord('s') and result:
                ts = int(time.time())
                snap_img = f"snapshot_{ts}.jpg"
                snap_json = f"snapshot_{ts}.json"
                cv2.imwrite(snap_img, display_frame)
                data = {
                    "timestamp": ts,
                    "primary_state": result.affect.primary_state,
                    "sincerity_score": result.affect.sincerity_score,
                    "is_masking_detected": result.affect.is_masking_detected,
                    "scientific_justification": result.affect.scientific_justification,
                    "layman_justification": result.affect.layman_justification,
                    "diagnostic_notes": result.affect.diagnostic_notes,
                    "gaze": {
                        "pitch_deg": result.gaze.avg_pitch,
                        "yaw_deg": result.gaze.avg_yaw,
                        "direction": result.gaze.gaze_direction,
                        "ear": result.gaze.avg_ear
                    },
                    "action_units": {
                        "au12_smile": result.aus.au12_lip_corner_puller,
                        "au6_cheek": result.aus.au6_cheek_raiser,
                        "au7_squint": result.aus.au7_lid_tightener,
                        "au1_grief": result.aus.au1_inner_brow_raiser,
                        "au4_furrow": result.aus.au4_brow_lowerer,
                        "duchenne_coherence": result.aus.duchenne_coherence
                    },
                    "pulse_bpm": result.pulse.bpm,
                    "emotion_radar": result.affect.emotion_radar
                }
                with open(snap_json, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"\n[SNAPSHOT SAVED] Frame: {snap_img} | Data: {snap_json}")

            elif key == ord('v'):
                # Toggle Video + Audio Recording
                if not is_recording_av:
                    # START RECORDING
                    is_recording_av = True
                    rec_start_time = time.time()
                    h, w, _ = display_frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(temp_vid_path, fourcc, 30.0, (w, h))
                    audio_rec.start()
                    print("\n🔴 [RECORDING STARTED]: Capturing Live Video + Microphone Audio...")
                else:
                    # STOP RECORDING
                    is_recording_av = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    audio_rec.stop(temp_aud_path)
                    final_rec_path = f"recording_{int(time.time())}.mp4"
                    print("\n⏹️ [RECORDING STOPPED]: Merging Video and Audio with FFmpeg...")

                    # Multiplex video and audio using ffmpeg if available
                    if os.path.exists(temp_aud_path) and os.path.exists(temp_vid_path):
                        cmd = [
                            "ffmpeg", "-y",
                            "-i", temp_vid_path,
                            "-i", temp_aud_path,
                            "-c:v", "copy",
                            "-c:a", "aac",
                            "-shortest",
                            final_rec_path
                        ]
                        try:
                            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                            print(f"🎉 Synchronized Video + Mic Audio saved to: {final_rec_path}")
                            if os.path.exists(temp_vid_path): os.remove(temp_vid_path)
                            if os.path.exists(temp_aud_path): os.remove(temp_aud_path)
                        except Exception as ex:
                            print(f"FFmpeg muxing note: {ex}. Video saved to {temp_vid_path}")
                    else:
                        print(f"Video saved to {temp_vid_path}")

    finally:
        if video_writer:
            video_writer.release()
        if audio_rec.is_recording:
            audio_rec.stop(temp_aud_path)
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()
        print("Live session terminated.")


def main():
    parser = argparse.ArgumentParser(description="Real-Time AffectSense Biometric HUD")
    parser.add_argument("--source", type=str, default=None, help="Image, video file, or camera index")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--key", type=str, default=None, help="Access Key or Admin Key to authorize session")
    parser.add_argument("--no-mesh", action="store_true", help="Hide landmark contours")
    parser.add_argument("--no-rays", action="store_true", help="Hide 3D retinal gaze rays")
    args = parser.parse_args()

    src = args.source if args.source is not None else args.camera
    run_live(source=src, show_mesh=not args.no_mesh, show_rays=not args.no_rays, key=args.key)



if __name__ == "__main__":
    main()
