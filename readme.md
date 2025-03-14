# AR Mirror

This project is an Augmented Reality (AR) Mirror application using Flask, OpenCV, and Mediapipe. It detects faces and hands in real-time video feed and overlays a GIF on an open palm.

## Requirements

- Python 3.6+
- Flask
- OpenCV
- Mediapipe
- Imageio

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/ar_mirror_python.git
   cd ar_mirror_python
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Place your overlay GIF in the `overlay` directory and update the `Settings` in `settings.py` if necessary.

## Usage

1. Run the Flask application:
   ```sh
   python app.py
   ```

2. Open your web browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## Settings

You can adjust the settings in `settings.py` to change the detection confidence, overlay size, and position.

## License

This project is licensed under the MIT License.