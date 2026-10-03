# YOLO Object Detection Web App

Browser-based version of the original YOLOv8 object detection project.

## Features
- Image upload and object detection
- Video upload and object detection
- People, bicycle, car and motorcycle counters
- Download detected image
- Download detected video
- No Tkinter or desktop `cv2.imshow()` required

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `yolov8n.pt`
   - `LICENSE`
   - `README.md`
3. Open https://share.streamlit.io/
4. Sign in with GitHub.
5. Click **Create app**.
6. Select your repository and branch.
7. Set the main file to `app.py`.
8. Choose a custom app subdomain if available.
9. Deploy.

Your final application will have a shareable `https://....streamlit.app` link.

## Important

The original `main.py` is kept for the desktop version. The live website uses `app.py`.

For very long/high-resolution videos, processing can take time because YOLO is running on CPU on a typical free cloud instance.
