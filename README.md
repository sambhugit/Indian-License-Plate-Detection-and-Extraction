# Indian-License-Plate-Detection-and-Extraction
AI tool to detect Indian License plate from videos(recorded and realtime) and extracts the license number using OCR. 
###
OCR detection is done using **contour segmantation**.
###
To use this repo,

1. Firstly, clone the repo to your workplace.
2. Then run,
 ```
 pip install -r requirements.txt 
 ``` 
3. Add the video on which detection is to be performed to same directory. 
4. To simply test the model, run 'inference.py' by providing 'path-to-video' in the main function.
5. To train the model unzip data.zip and extract it to the same directory. In Colab or Jupyter, simply run
```
!unzip data.zip
```
6. Run 'train.py'!
