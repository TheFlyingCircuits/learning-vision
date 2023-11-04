# learning-vision

This Python script is designed to perform real-time image processing using a webcam feed. It also has the capability to switch to processing a static image.

The purpose of this script is so programmers can work on developing Limelight python pipelines
without access to a physical Limelight.

## Requirements

Before getting started, you need to set up your Python environment:
Make sure you have Python installed on your system. You can download it from the [official website](https://www.python.org/downloads/).

To run this script, you'll need to install the necessary dependencies. These are listed in the [`requirements.txt`](requirements.txt) file.

## Setup Instructions

1. **Clone the repository to your local machine**:

   ```bash
   git clone https://github.com/TheFlyingCircuits/learning-vision.git
   ```

2. **Install required dependancies**:

   ```bash
   cd learning-vision
   pip install -r requirements.txt
   ```

3. **Place the Static Image**: If you plan to use the 'pic' mode, place the static image you wish to use in the same directory as the script, and ensure it is named [`image.png`](image.png).

## Usage

To run the script, use the following command in your terminal:

```bash
python vision.py
```

## Key Controls

- `q`: Quit the application.
- `z`: Set camera resolution to 320x240 and 90 FPS.
- `x`: Set camera resolution to 640x480 and 90 FPS.
- `c`: Set camera resolution to 960x720 and 22 FPS.
- `v`: Set camera resolution to 1280x960 and 22 FPS.
- `0-9`: Select different processing pipelines. `0` and any unused numbers currently results in no processing
- `Spacebar`: Toggle between camera input and static image.

### Note

> Make sure your webcam supports the resolutions and frame rates specified in the script. If the requested resolution or FPS is not supported by your hardware, the webcam may default to the closest supported settings or not work as intended.

## Extending

To extend the application with new pipelines:

1. Add a new Python script in the `runPipeline` directory.
2. Define a `runPipeline` function within the script with the appropriate image processing code.
3. Import the script and add a new `elif` condition in the `runPipeline.runPipeline` function to include your new pipeline.

## Support

For support, open an issue in the repository.

## Contributing

Contributions are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
