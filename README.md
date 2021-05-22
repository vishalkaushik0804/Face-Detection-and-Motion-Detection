# Face-Detection-and-Motion-Detection
1. **Face Detection**:
        
        *Haar Cascade* is an object detection algorithm. It can be used to detect objects in images or videos. 
        
        The algorithm has four stages:
        
            1. Haar Feature Selection
            2. Creating  Integral Images
            3. Adaboost Training
            4. Cascading Classifiers
            
2. **Motion Detection**:
        
        *Motion Detector* will allow you to detect motion and also store the time interval of the motion.
        Videos can be treated as stack of pictures called frames. 
        Different frames(pictures) are compared to the first frame which should be static(No movements initially). 
        We compare two images by comparing the intensity value of each pixels.
        Also, we store the time interval of motion in a CSV file. 
        
        Motion Detector has four Windows:
        
            1. Gray Frame
            2. Difference Frame
            3. Threshold Frame
            4. Color Frame
