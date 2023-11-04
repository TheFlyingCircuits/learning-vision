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
- `Enter`: Dynamically reloads all preexisting pipelines to allow for code changes without restarting.

### Note

> Make sure your webcam supports the resolutions and frame rates specified in the script. If the requested resolution or FPS is not supported by your hardware, the webcam may default to the closest supported settings or not work as intended.

## Extending

To extend the application with new pipelines:

1. Add a new Python script in the `pipelines` directory.
2. Define a `runPipeline` function within the script with the appropriate image processing code.
3. Import the script and add a new `elif` condition in the `pipelineManager.manage` function to include your new pipeline.
4. Add a tuple to the `modules_to_reload` list in `pipelineManager.reloadPipelines` that contains the module object and it's string name.

## Dynamic Reload Feature

The dynamic reload feature in our application allows you to update the image processing pipelines without restarting the entire script. This means you can make changes to your pipeline code and apply them on-the-fly, which is particularly useful during development and testing.

### How to Use Dynamic Reload

To use the dynamic reload feature, follow these simple steps:

1. Make your desired code changes in the pipeline modules.
2. Focus on the main application window.
3. Press the `Enter` key to trigger the reload process.

The application will attempt to reload all pipeline modules and print the status of each reload in the console:

- A message `Reloaded [module_name] successfully.` indicates a successful reload.
- A message `Failed to reload [module_name]: [error_message]` indicates that there was an error during the reload of the specified module.

### Error Handling

If any of the pipeline modules fail to reload due to errors in the code, the feature will catch these exceptions and prevent the entire application from crashing. This way, you can simply fix the issue in the relevant module and try reloading again.

### Benefits of Dynamic Reload

- **Efficiency**: Eliminates the need for restarting the script to apply changes, saving time.
- **Continuous Feedback**: Instantly test and debug your pipeline modifications.
- **Stability**: Isolates errors to individual modules, maintaining overall application stability.

### Important Notes

- The dynamic reload feature does not reload the entire state of the script. Variables and objects outside of the module scope will retain their state from before the reload.

## Support

For support, open an issue in the repository.

## Contributing

Contributions are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
