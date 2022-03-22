import os
import sys
import cv2
import numpy as np
import paddle
import math
import time
import collections

from openvino.runtime import Core,PartialShape,Dimension
from IPython import display
import copy

sys.path.append("../utils")
import notebook_utils as utils
from pre_post_processing import *
from IPython import display

det_model_dir = "./model/ch_ppocr_mobile_v2.0_det_infer"
det_model_file_path = det_model_dir + "/inference.pdmodel"
det_params_file_path = det_model_dir + "/inference.pdiparams"

# initialize inference engine for text detection
det_ie = Core()
det_net = det_ie.read_model(model=det_model_file_path, weights=det_params_file_path)
det_compiled_model = det_ie.compile_model(model=det_net, device_name="CPU")

# get input and output nodes for text detection
det_input_layer = next(iter(det_compiled_model.inputs))
det_output_layer = next(iter(det_compiled_model.outputs))

rec_model_dir = "./model/ch_ppocr_mobile_v2.0_rec_infer"
rec_model_file_path = rec_model_dir + "/inference.pdmodel"
rec_params_file_path = rec_model_dir + "/inference.pdiparams"

 # Initialize the Paddle recognition inference on CPU
rec_ie = Core()
# read the model and corresponding weights from file
rec_net = rec_ie.read_model(model=rec_model_file_path, weights=rec_params_file_path)

# assign dynamic shapes to every input layer on the last dimension
for input_layer in rec_net.inputs:
    input_shape = input_layer.partial_shape
    input_shape[3] = Dimension(-1)
    rec_net.reshape({input_layer: input_shape})

rec_compiled_model = rec_ie.compile_model(model=rec_net, device_name="CPU")

# get input and output nodes
rec_input_layer = next(iter(rec_compiled_model.inputs))
rec_output_layer = next(iter(rec_compiled_model.outputs))

def image_preprocess(input_image, size):
    img = cv2.resize(input_image, (size,size))
    img = np.transpose(img, [2,0,1]) / 255
    img = np.expand_dims(img, 0)
    ##NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    img_mean = np.array([0.485, 0.456,0.406]).reshape((3,1,1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
    img -= img_mean
    img /= img_std
    return img.astype(np.float32)

def resize_norm_img(img, max_wh_ratio):
        rec_image_shape = [3, 32, 320]
        imgC, imgH, imgW = rec_image_shape
        assert imgC == img.shape[2]
        character_type = "ch"
        if character_type == "ch":
            imgW = int((32 * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

def run_paddle_ocr(source=0, flip=False, use_popup=False, skip_first_frames=10):
    # create video player to play with target fps
    player = None
    try:
        player = utils.VideoPlayer(source=source, flip=flip, fps=10, skip_first_frames=skip_first_frames)
        #Start video capturing
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        processing_times = collections.deque()
        det_request = det_compiled_model.create_infer_request()
        while True:
            # grab the frame
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            # if frame larger than full HD, reduce size to improve the performance
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(src=frame, dsize=None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            else:    
                #Filp the image otherwise the recognition result is wrong
                image_file = cv2.flip(frame,1)
                test_image = image_preprocess(image_file,640)

                # measure processing time for text detection
                start_time = time.time()
                #perform the inference step
                det_request.infer(inputs={det_input_layer.any_name: test_image})
                det_results = det_request.get_tensor(det_output_layer).data
                stop_time = time.time()

                # Postprocessing for Paddle Detection
                ori_im = image_file.copy()
                data = {'image': image_file}
                data_resize = DetResizeForTest(data)
                data_norm = NormalizeImage(data_resize)
                data_list = []
                keep_keys =  ['image', 'shape']
                for key in keep_keys:
                    data_list.append(data[key])
                img, shape_list = data_list

                shape_list = np.expand_dims(shape_list, axis=0) 
                pred = det_results[0]    
                if isinstance(pred, paddle.Tensor):
                    pred = pred.numpy()
                segmentation = pred > 0.3

                boxes_batch = []
                for batch_index in range(pred.shape[0]):
                    src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
                    mask = segmentation[batch_index]
                    boxes, scores = boxes_from_bitmap(pred[batch_index], mask,src_w, src_h)
                    boxes_batch.append({'points': boxes})
                post_result = boxes_batch
                dt_boxes = post_result[0]['points']
                dt_boxes = filter_tag_det_res(dt_boxes, ori_im.shape)
                #Draw boxes on detected text locations
                src_im = draw_text_det_res(dt_boxes, image_file)

                processing_times.append(stop_time - start_time)
                # use processing times from last 200 frames
                if len(processing_times) > 200:
                    processing_times.popleft()
                processing_time_det = np.mean(processing_times) * 1000

                #Preprocess detection results for recognition
                dt_boxes = sorted_boxes(dt_boxes)
                img_crop_list = []   
                if dt_boxes != []:
                    for bno in range(len(dt_boxes)):
                        tmp_box = copy.deepcopy(dt_boxes[bno])
                        img_crop = get_rotate_crop_image(ori_im, tmp_box)
                        img_crop_list.append(img_crop)

                    #Recognition starts from here
                    img_num = len(img_crop_list)
                    # Calculate the aspect ratio of all text bars
                    width_list = []
                    for img in img_crop_list:
                        width_list.append(img.shape[1] / float(img.shape[0]))
                    # Sorting can speed up the recognition process
                    indices = np.argsort(np.array(width_list))
                    rec_res = [['', 0.0]] * img_num
                    batch_num = 6
                    rec_processing_times = 0

                    #For each detected text box, run inference for text recognition
                    for beg_img_no in range(0, img_num, batch_num):
                        end_img_no = min(img_num, beg_img_no + batch_num)

                        norm_img_batch = []
                        max_wh_ratio = 0
                        for ino in range(beg_img_no, end_img_no):
                            h, w = img_crop_list[indices[ino]].shape[0:2]
                            wh_ratio = w * 1.0 / h
                            max_wh_ratio = max(max_wh_ratio, wh_ratio)
                        for ino in range(beg_img_no, end_img_no):
                            norm_img = resize_norm_img(img_crop_list[indices[ino]],max_wh_ratio)
                            norm_img = norm_img[np.newaxis, :]
                            norm_img_batch.append(norm_img)

                        norm_img_batch = np.concatenate(norm_img_batch)
                        norm_img_batch = norm_img_batch.copy()

                        #Run inference for text recognition 
                        for index in range(len(norm_img_batch)):
                            rec_request = rec_compiled_model.create_infer_request()
                            rec_request.infer(inputs={rec_input_layer.any_name: norm_img_batch})
                            rec_results = rec_request.get_tensor(rec_output_layer).data
                        preds = rec_results
    
                        #Postprocessing recognition results
                        postprocess_op = build_post_process(postprocess_params)
                        rec_result = postprocess_op(preds)
                        for rno in range(len(rec_result)):
                            rec_res[indices[beg_img_no + rno]] = rec_result[rno]

                    #Text recognition results, rec_res, include two parts:
                    #txts are the recognized text results, scores are the recognition confidence level                   
                    if rec_res != []:
                        image = Image.fromarray(cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB))
                        boxes = dt_boxes
                        txts = [rec_res[i][0] for i in range(len(rec_res))] 
                        scores = [rec_res[i][1] for i in range(len(rec_res))] 
                        
                        #draw text recognition results beside the image
                        draw_img = draw_ocr_box_txt(
                                    image,
                                    boxes,
                                    txts,
                                    scores,
                                    drop_score=0.5)
                        
                        #Visualize PPOCR results
                        _, f_width = draw_img.shape[:2]
                        fps = 1000 / processing_time_det
                        cv2.putText(img=draw_img, text=f"OpenVINO Inference time: {processing_time_det:.1f}ms ({fps:.1f} FPS)", org=(20, 40),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=f_width / 1000,
                                color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                        
                        # use this workaround if there is flickering
                        if use_popup: 
                            draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
                            cv2.imshow(winname=title, mat=draw_img)
                            key = cv2.waitKey(1)
                            # escape = 27
                            if key == 27:
                                break
                        else:
                            # encode numpy array to jpg
                            _, encoded_img = cv2.imencode(ext=".jpg", img=draw_img,
                                                                params=[cv2.IMWRITE_JPEG_QUALITY, 100])
                            # create IPython image
                            i = display.Image(data=encoded_img)
                            # display the image in this notebook
                            display.clear_output(wait=True)
                            display.display(i)
            
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # stop capturing
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()
video_file = "./data/test_video.mp4"
run_paddle_ocr(source=0, flip=True, use_popup=True)