class Config:

    # Camera Settings
    cameraIndex = 0
    frameWidth = 960
    frameHeight = 700
    minDetectionConfidence = 0.5

    # YOLO Settings
    modelPath = "yolov8n.pt"
    enableGPU = False # <- Set to True when testing on something else bedies my laptop with a CUDA-capable GPU

    # Display Settings
    showLivePreview = True
    previewWindowName = "Luminaris Security Monitor"

    # Detection Classes (From the COCO dataset class IDs)
    detectAllClasses = True # <- Set to false when testing for security objects right now its true for general testing
    # securityClassIDs = [0, 2, 3, 5, 7]
    maxDetectionsPerFrame = 15

    # Debug Settings
    printDetections = False #<- Set to false to reduce the console outputs
    showFPS = True