import cv2
import math
import numpy as np
from ultralytics import YOLO
from seaborn import color_palette
import os
import torch


def load_class_names(file_name):
    """
    Returns a list of class names read from the file `file_name`.

    Args:
        file_name (str): The path to the file containing the class names.

    Returns:
        List[str]: A list of class names.
    """

    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names


def draw_bbox(frame, boxes, class_names, colors):
    """
    Draws bounding boxes with labels on the input frame.

    Args:
        frame (numpy.ndarray): The input image frame.
        boxes (List[Object]): List of bounding boxes.
        class_names (List[str]): List of class names.
        colors (List[Tuple[int]]): List of RGB color values.

    Returns:
        None
    """
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Extracting the class label and name
        cls = int(box.cls[0])
        class_name = class_names[cls]

        # Retrieving the color for the class
        color = colors[cls]

        # Drawing the bounding box on the image
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Formatting the confidence level and label text
        conf = math.ceil((box.conf[0] * 100)) / 100
        label = f'{class_name} ({conf}%)'

        # Calculating the size of the label text
        text_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        # Calculating the coordinates for the background rectangle of the label
        rect_coords = x1 + text_size[0], y1 - text_size[1] - 3

        # Drawing the background rectangle and the label text
        cv2.rectangle(frame, (x1, y1), rect_coords, color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, y1 - 2), 0, 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


def run_yolo(model_name='yolo_assets/Models/pcbdetect.pt', source=0, prediction_type='video',
             class_path="yolo_assets/Classes/classes.txt",
             outdir='yolo_assets/Detections/output',
             web_app=False):
    """
    Performs object detection on an image or video.

    Args:
        model_name (str): The name of the model to use for object detection. Default is 'yolov8s.pt'.
        source (Union[str, int]): The path to the image or video file or webcam index. Default is 0 (webcam).
        prediction_type (str): The type of prediction to make. Valid values are 'image' and 'video'. Default is 'video'.
        class_path (str): The path to the file containing class names. Default is 'classes.txt'.
        outdir (str): The output directory or file name. Default is 'output'.
        web_app (bool): Whether the output is for a web application. Default is False.

    Returns:
        If `web_app` is True and `prediction_type` is 'video', it yields each frame with object detection results.
    """
    # Initializing the YOLO model
    model = YOLO(model_name)
    if torch.cuda.is_available():
        model.cuda()
    output_directory = os.path.dirname(outdir)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Loading the class names from the file
    class_names = load_class_names(class_path)
    n_classes = len(class_names)
    #  Generating colors for each class
    colors = {}
    #  Generate a color palette
    for i in range(n_classes):
        color = tuple((np.array(color_palette('hls', n_classes)) * 255)[i])
        colors[i] = color

    # Checking the prediction type
    if prediction_type == 'video':
        # Capturing the video from the source
        cap = cv2.VideoCapture(source)

        if not web_app:
            # Appending '.avi' extension to the output directory
            outdir = outdir + '.avi'
            # Getting the frame width and height
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            # Creating a VideoWriter object to save the output
            out = cv2.VideoWriter(outdir, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

        while True:
            # Reading a frame from the video
            ret, frame = cap.read()

            # Performing object detection on the frame
            results = model(frame, stream=True, conf=0.1, verbose=False)

            # Iterating over the detected objects
            for i, result in enumerate(results):
                # Extracting the bounding box coordinates
                boxes = result.boxes
                draw_bbox(frame, boxes, class_names, colors)

            if not web_app:
                # Writing the modified frame to the output video file
                out.write(frame)

            # Displaying the image if web_app is False
            if not web_app:
                cv2.imshow("Image", frame)

                # Checking if the user pressed 'q' to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Yielding the frame if web_app is True
            if web_app:
                yield frame

        # Releasing the VideoWriter and closing the window
        out.release()

    elif prediction_type == 'image':
        # Appending '.jpg' extension to the output directory
        outdir = outdir + '.jpg'
        frame = cv2.imread(source)
        results = model(frame, stream=True, conf=0.5)

        # Iterating over the detected objects
        for i, result in enumerate(results):
            # Extracting the bounding box coordinates
            boxes = result.boxes
            draw_bbox(frame, boxes, class_names, colors)
            cv2.imshow("Image", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if not web_app:
            cv2.imwrite(outdir, frame)


if __name__ == '__main__':

    # Run yolo on image
    func = run_yolo(source='yolo_assets/Data/ppe-2.mp4', prediction_type='video')
    # Run yolo on video
    # func = run_yolo(source='yolo_assets/Data/video.mp4', prediction_type='video')
    # Run yolo on an camera
    # func = run_yolo()

    # Consume the generator to avoid unexpected behavior
    for frame in func:
        pass
