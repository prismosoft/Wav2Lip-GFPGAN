from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
from face_parsing import init_parser, swap_regions
import platform
import cv2

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--segmentation_path', type=str, help='Name of saved checkpoint of segmentation network',
                    required=True)

parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', default='results/result_voice.mp4', type=str,
                    help='Video path to save result. See default for an e.g.')

parser.add_argument('--static', default=False, action='store_true',
                    help='If True, then use only first video frame for inference')
parser.add_argument('--fps', type=float, default=25.,
                    help='Can be specified only if input is a static image (default: 25)', required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, default=16, help='Batch size for face detection')
parser.add_argument('--wav2lip_batch_size', type=int, default=128, help='Batch size for Wav2Lip model(s)')

parser.add_argument('--resize_factor', default=1, type=int,
                    help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected. Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg. Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')

parser.add_argument('--no_segmentation', default=False, action='store_true', help='Prevent using face segmentation')

parser.add_argument('--rembg', default=False, action='store_true', help='Remove background')

parser.add_argument('--fr', default=False, action='store_true', help='Use face restoration')
parser.add_argument('--fr_scale', type=int, default=2, help='Face restoration scale')
parser.add_argument('--fr_version', default=1.4, help='Face restoration scale')

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    args.static = True


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    batch_size = args.face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results


def datagen(mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    """
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
    """
    reader = read_frames()
    for i, m in enumerate(mels):
        try:
            frame_to_save = next(reader)
        except StopIteration:
            reader = read_frames()
            frame_to_save = next(reader)
        face, coords = face_detect([frame_to_save])[0]

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def read_frames():
    if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        face = cv2.imread(args.face)
        while 1:
            yield face
    video_stream = cv2.VideoCapture(args.face)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    print('Reading video frames from start...')
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if args.resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))
        if args.rotate:
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
        y1, y2, x1, x2 = args.crop
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]
        frame = frame[y1:y2, x1:x2]
        yield frame


def face_restoration(removebg=False, removebg_only=False):
    from rembg import remove

    current_path = os.getcwd()
    parent_path = os.path.abspath(os.path.join(current_path, os.pardir))

    gfgan_path = os.path.join(parent_path, 'GFPGAN-master')
    temp_folder = os.path.join(current_path, 'temp')
    unprocessed_frames_path = os.path.join(temp_folder, 'frames')
    restored_frames_path = os.path.join(temp_folder, 'restored_imgs')
    concat_file_path = os.path.join(temp_folder, 'concat.txt')
    concated_video_output = os.path.join(temp_folder, 'concated_output.avi')
    temp_video_path = os.path.join(temp_folder, 'result.avi')
    temp_restored_video_path = os.path.join(temp_folder, 'restored_result.avi')

    if not os.path.exists(unprocessed_frames_path):
        os.makedirs(unprocessed_frames_path)

    if not os.path.exists(restored_frames_path):
        os.makedirs(restored_frames_path)

    vidcap = cv2.VideoCapture(temp_video_path)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print("FPS: ", fps, "Frames: ", frame_count)

    for frameNumber in tqdm(range(frame_count)):
        _, image = vidcap.read()

        if removebg:
            image = remove(image)

        cv2.imwrite(path.join(unprocessed_frames_path, str(frameNumber).zfill(4) + '.png'), image)

    if not removebg_only:
        command = 'python ' + gfgan_path + '/inference_gfpgan.py \
                    -i ' + unprocessed_frames_path + ' \
                    -o ' + temp_folder + ' \
                    -v ' + str(args.fr_version) + ' \
                    -s ' + str(args.fr_scale) + ' \
                    --only_center_face \
                    --bg_upsampler None'

        subprocess.call(command, shell=True)
    else:
        restored_frames_path = unprocessed_frames_path

    dir_list = os.listdir(restored_frames_path)
    dir_list.sort()

    batch = 0
    batch_size = 300

    for i in tqdm(range(0, len(dir_list), batch_size)):
        img_array = []
        start, end = i, i + batch_size
        print("processing ", start, end)
        for filename in tqdm(dir_list[start:end]):
            filename = restored_frames_path + '/' + filename;
            img = cv2.imread(filename)
            if img is None:
                continue
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(temp_folder + '/batch_' + str(batch).zfill(4) + '.avi',
                              cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        batch = batch + 1

        for im in range(len(img_array)):
            out.write(img_array[im])
        out.release()

    concat_file = open(concat_file_path, "w")
    for ips in range(batch):
        print(ips)
        concat_file.write("file batch_" + str(ips).zfill(4) + ".avi\n")
    concat_file.close()

    command = 'ffmpeg -y -f concat -i ' + concat_file_path + ' -c copy ' + concated_video_output
    subprocess.call(command, shell=True)

    return temp_restored_video_path

def main():
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        #full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        video_stream.release()

    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    batch_size = args.wav2lip_batch_size
    gen = datagen(mel_chunks)

    enumerated = enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size))))

    for i, (img_batch, mel_batch, frames, coords) in enumerated:
        if i == 0:
            print("Loading segmentation network...")
            seg_net = init_parser(args.segmentation_path)

            model = load_model(args.checkpoint_path)
            print("Model loaded")

            frame_h, frame_w = next(read_frames()).shape[:-1]

            out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c

            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            if not args.no_segmentation:
                p = swap_regions(f[y1:y2, x1:x2], p, seg_net)

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    video_result = 'temp/result.avi'

    if args.fr:
        video_result = face_restoration(args.rembg)
    else:
        if args.rembg:
            video_result = face_restoration(True, True)

    command = 'ffmpeg -y -i {} -i {} -vcodec libx264 -vprofile high -crf 28 {}'.format(args.audio, video_result, args.outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
    main()
