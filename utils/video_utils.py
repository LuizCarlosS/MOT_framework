import cv2


def read_video(path):
    """
    Read a video from the given path.
    """
    return cv2.VideoCapture(path)

def get_video_info(video):
    """
    Get information about the given video, such as its height, width, and frame count.
    """
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    return height, width, frame_count, fps

def read_frame(video):
    """
    Read a frame from the given video.
    """
    ret, frame = video.read()
    if not ret:
        return None
    return frame

def write_frame(video_writer, frame):
    """
    Write a frame to the given video writer.
    """
    video_writer.write(frame)

def release_video(video):
    """
    Release the given video.
    """
    video.release()
