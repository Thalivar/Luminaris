import cv2, time
from core.camera import CameraManager
from detection.yoloDetector import YOLODetector
from config.settings import Config

def main():
    print("Initializing Luminaris Security System...")

    camera = CameraManager()
    detector = YOLODetector()

    if not camera.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    fpsCounter = 0
    startTime = time.time()

    try:
        while True:
            frame = camera.readFrame()
            if frame is None:
                print("Failed to read frame from the camerea")
                break

            detections = detector.detectObjects(frame)
            frameWithDetections = detector.drawDetections(frame, detections)
            fpsCounter += 1

            elapsedTime = time.time() - startTime
            if elapsedTime >= 1.0:
                fps = fpsCounter / elapsedTime
                fpsCounter = 0
                startTime = time.time()
            else:
                fps = fpsCounter / max(elapsedTime, 0.01)
            
            infoText = f"FPS: {fps:.1f} | Detections: {len(detections)}"
            cv2.putText(frameWithDetections, infoText, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if detections:
                print(f"Frame {camera.frameCount}: Found {len(detections)} objects")
                for detection in detections:
                    print(f" - {detection['className']}: {detection['confidence']:.2f}")
            
            if Config.showLivePreview:
                cv2.imshow(Config.previewWindowName, frameWithDetections)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    except KeyboardInterrupt:
        print("\nShutdown Requested by user")
    except Exception as e:
        print(f"Error occured: {e}")
    
    finally:
        print("Cleaning up...")
        camera.release()
        cv2.destroyAllWindows()
        print("Luminaris shutdow complete")

if __name__ == "__main__":
    main()