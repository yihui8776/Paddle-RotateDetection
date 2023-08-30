# 代码示例
# python predict.py [src_image_dir] [results]

import os
import sys
import glob
import json
import cv2
import numpy as np
import paddle
import yaml
from preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride, LetterBoxResize, WarpAffine, Pad, decode_image
import math
import xml.etree.ElementTree as ET
from collections import defaultdict

roannosdir = "/home/aistudio/work/roAnns2"
if not os.path.exists(roannosdir):
    os.makedirs(roannosdir)
# 打开json标签文件
with open(r'/home/aistudio/PaddleDetection/dataset/gzp/annotations/train.json','r') as f:
    data_an = {}  # 外围大字典
    json_dicts = json.loads(f.read())  #

cate = json_dicts['categories']
print(cate)
cls_name = [0]*(len(cate)+1)
print(cls_name)
for cate_dict in cate:
    cls_name[cate_dict['id']] = cate_dict['name'] # 各个类别对应的名字 按顺序放在cls_name中，如id为0的类别名就是cls_name[0]


def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of images (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    inputs = {}

    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs['image'] = np.array((imgs[0], )).astype('float32')
        inputs['im_shape'] = np.array(
            (im_info[0]['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info[0]['scale_factor'], )).astype('float32')
        return inputs

    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

    inputs['im_shape'] = np.concatenate(im_shape, axis=0)
    inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs['image'] = np.stack(padding_imgs, axis=0)
    return inputs

def preprocess_mine(image_list,preconfig):
        preprocess_ops = []
        for op_info in preconfig:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []
        for im_path in image_list:
            im, im_info = preprocess(im_path, preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        return inputs

def preconfig():
    deploy_file = os.path.join('/home/aistudio/PaddleDetection/inference_model/gzp-v2/infer_cfg.yml')
    with open(deploy_file) as f:
        yml_conf = yaml.safe_load(f)
    return yml_conf['Preprocess']


def coordinate_present_convert(coords, shift=True):
    """
    :param coords: shape [-1, 5]
    :param shift: [-90, 90) --> [-180, 0)
    :return: shape [-1, 5]
    """
    # angle range from [-90, 0) to [0,180)
    w, h = coords[:, 2], coords[:, 3]

    remain_mask = np.greater(w, h)
    convert_mask = np.logical_not(remain_mask).astype(np.int32)
    remain_mask = remain_mask.astype(np.int32)

    remain_coords = coords * np.reshape(remain_mask, [-1, 1])

    coords[:, [2, 3]] = coords[:, [3, 2]]
    coords[:, 4] += 90

    convert_coords = coords * np.reshape(convert_mask, [-1, 1])

    coords_new = remain_coords + convert_coords


    if shift:
        if coords_new[:, 4] >= 0:
            coords_new[:, 4] = 180 + coords_new[:, 4]

    return np.array(coords_new, dtype=np.float32)

def backward_convert(coordinate):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: format [x_c, y_c, w, h, theta, (label)]
    """
    boxes = []

    box = np.int0(coordinate)
    box = box.reshape([4, 2])
    rect1 = cv2.minAreaRect(box)

    x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]

    if theta == 0:
        w, h = h, w
        theta -= 90

    boxes.append([x, y, w, h, theta])

    return np.array(boxes, dtype=np.float32)


def get_rotation(cords):

    # 用OpenCV的最小矩形转换成-90到0的角度
    boxes = backward_convert(cords )
    # 根据长短边转成 0 到 180
    new_boxes = coordinate_present_convert(boxes)
    # 转成弧度：
    new_boxes[0][-1] = new_boxes[0][-1] * math.pi/180
    #new_boxes = new_boxes.astype(np.str)
    print(new_boxes)
    cx,cy,w,h,angle = new_boxes[0]
    if angle<0:
        angle = 3.141593+angle
    return cx,cy,w,h,angle


def polygon_area(poly):
    #计算多边形的面积
    #[x1,y1,x2,y2,x3,y3,x4,y4,prob,label,cx,cy,w,h,angle]
    #polys = np.array[(poly[0],poly[1]),(poly[2],poly[3]),(poly[4],poly[5]),(poly[6],poly[7])]
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_iou(poly1, poly2):
    #两个多边形的交并比
    #polys1 = np.array[(poly1[0],poly1[1]),(poly1[2],poly1[3]),(poly1[4],poly1[5]),(poly1[6],poly1[7])]
    #polys2 = np.array[(poly2[0],poly2[1]),(poly2[2],poly2[3]),(poly2[4],poly2[5]),(poly2[6],poly2[7])]
    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)
    
    intersection_poly = np.concatenate((poly1, poly2))
    min_x, min_y = np.min(intersection_poly, axis=0)
    max_x, max_y = np.max(intersection_poly, axis=0)
    
    grid = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
    
    poly1[:, 0] -= min_x
    poly1[:, 1] -= min_y
    poly2[:, 0] -= min_x
    poly2[:, 1] -= min_y
    
    cv2.fillPoly(grid, [poly1.astype(np.int32)], 1)
    cv2.fillPoly(grid, [poly2.astype(np.int32)], 2)

    intersection_area = np.sum(grid == 3)
    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area
    return iou
 
def nms_polygons(polygons, iou_threshold):
    #多边形nms  
    #输入多边形列表 [(x1, y1), (x2, y2), ..., (xn, yn), probability]
    #[x1,y1,x2,y2,x3,y3,x4,y4,prob,label,cx,cy,w,h,angle]
    selected_polygons = []
    
    polygons.sort(key=lambda x: x[8], reverse=True)  # 按概率值降序排列

    while len(polygons) > 0:
        current_polygon = polygons.pop(0)
        selected_polygons.append(current_polygon)

        polygons_to_remove = []
        for polygon in polygons:
            currentpoly = np.array([(current_polygon[0],current_polygon[1]),(current_polygon[2],current_polygon[3]),(current_polygon[4],current_polygon[5]),(current_polygon[6],current_polygon[7])])
            poly = np.array([(polygon[0],polygon[1]),(polygon[2],polygon[3]),(polygon[4],current_polygon[5]),(polygon[6],polygon[7])])
            iou = polygon_iou(currentpoly, poly)
            if iou > iou_threshold:
                polygons_to_remove.append(polygon)

        for polygon_to_remove in polygons_to_remove:
            polygons.remove(polygon_to_remove)

    return selected_polygons


def process(src_image_dir, save_dir):
    paddle.device.set_device('gpu:0')
    # 创建 config
    # config = paddle_infer.Config("./model/mine/model.pdmodel", "./model/mine/model.pdiparams")

    # # 根据 config 创建 predictor
    # predictor = paddle_infer.create_predictor(config)

    # # 获取输入 Tensor
    # input_names = predictor.get_input_names()
    # input_tensor = predictor.get_input_handle(input_names[0])
    # print(input_names)
    # print(input_tensor)
    # exit()
    model = paddle.jit.load('/home/aistudio/PaddleDetection/inference_model/gzp-v1/model')
    model.eval()
    preconfigs = preconfig()
    iou_threshold = 0.5
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))

    results = {}
    result = defaultdict(list)
    for image_path in image_paths:
        img = cv2.imread(image_path)
        width = img.shape[1]
        height = img.shape[0]

        filename = os.path.split(image_path)[1]
        inputs = preprocess_mine([image_path], preconfigs)
        im_shape = inputs['im_shape'][0]
        image = inputs['image'][0]
        scale_factor = inputs['scale_factor'][0]

        #image = image[:,:int(im_shape[0]),:int(im_shape[1])]
        image = paddle.to_tensor(image).astype('float32')
        image = paddle.reshape(image,[1]+image.shape)

        scale_factor = paddle.to_tensor(scale_factor).reshape((1, 2)).astype('float32')
        im_shape = paddle.to_tensor(im_shape).reshape((1, 2)).astype('float32')
        #print(im_shape.shape)
        # print(image.shape)
        # print(scale_factor.shape)
        # exit()

        preds = model(image, scale_factor)
        preds = preds[0].numpy().tolist()
        #print(preds)
        # do something
        if filename not in result:
            predbox = []
           #selected = []
            result[filename] = []
            results[filename] = []
        for k in range(len(preds)):
            if k == 0 or preds[k][1] > 0.3: #概率
                #print(cls_name[int(preds[k][0]+1)])
                label = cls_name[int(preds[k][0]+1)]  #类别
                prob = preds[k][1]
                #print(preds[k][1])
                #print(preds[k])
                for j in range(len(preds[k])):
                    if j >= 2:
                        preds[k][j] = int(preds[k][j]) if preds[k][j] > 0 else 0
                        if j % 2 == 0:
                            preds[k][j] = int(preds[k][j]) if preds[k][j] < width else width
                        elif j % 2 == 1:
                            preds[k][j] = int(preds[k][j]) if preds[k][j] < height else height

                lb_x, lb_y = int(preds[k][2]), int(preds[k][3])
                lt_x, lt_y = int(preds[k][4]), int(preds[k][5])
                rt_x, rt_y = int(preds[k][6]), int(preds[k][7])
                rb_x, rb_y = int(preds[k][8]), int(preds[k][9])

                x1,y1 = lt_x,lt_y  #left top
                x2,y2 = rt_x,rt_y
                x3,y3 = rb_x,rb_y
                x4,y4 = lb_x,lb_y

                xmin = min(lb_x, lt_x, rt_x, rb_x)
                ymin = min(lb_y, lt_y, rt_y, rb_y)
                xmax = max(lb_x, lt_x, rt_x, rb_x)
                ymax = max(lb_y, lt_y, rt_y, rb_y)

                cx,cy,w,h ,angle = get_rotation([x1,y1,x2,y2,x3,y3,x4,y4])
                
                predbox.append([x1,y1,x2,y2,x3,y3,x4,y4,prob,label,cx,cy,w,h,angle])
                # selected = 
                # result[filename].append([label, x1, y1, x2, y2,x3,y3,x4,y4,cx, cy, w, h, angle])
                # results[filename].append({
                #     #"box":[xmin, ymin, xmax, ymax],
                #     "label": label,
                #     "x1": x1,
                #     "y1": y1,
                #     "x2": x2,
                #     "y2": y2,
                #     "x3": x3,
                #     "y3": y3,
                #     "x4": x4,
                #     "y4": y4,
                #     "cx": str(cx),
                #     "cy": str(cy),
                #     "w": str(w),
                #     "h": str(h),
                #     "angle": str(angle) 
                # })
        result[filename] = nms_polygons(predbox, iou_threshold)
        for res in result[filename]:
            results[filename].append({
                    #"box":[xmin, ymin, xmax, ymax],
                    "label": res[9],
                    "x1": res[0],
                    "y1": res[1],
                    "x2": res[2],
                    "y2": res[3],
                    "x3": res[4],
                    "y3": res[5],
                    "x4": res[6],
                    "y4": res[7],
                    "cx": str(res[10]),
                    "cy": str(res[11]),
                    "w": str(res[12]),
                    "h": str(res[13]),
                    "angle": str(res[14]) 
                })

        print(filename)
        with open(os.path.join(save_dir, filename.rstrip('.jpg')) + '.xml', 'w') as f:
            f.write(f"""<annotation>
            <folder>JPEGImages</folder>
            <filename>{filename}</filename>
            <size>
                <width>{width}</width>
                <height>{height}</height>
                <depth>3</depth>
            </size>
            <segmented>0</segmented>\n""")
            for x1, y1, x2, y2,x3,y3,x4,y4, prob,label,cx ,cy, w, h, angle in result[filename]:
                    f.write(f"""<object>
                <type>robndbox</type>
                <name>{label}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <robndbox>
                  <cx>{cx}</cx>
                  <cy>{cy}</cy>
                  <w>{w}</w>
                  <h>{h}</h>
                  <angle>{angle}</angle>
                </robndbox>
                <segmentation>
                  <x1>{x1}</x1>
                  <y1>{y1}</y1>
                  <x2>{x2}</x2>
                  <y2>{y2}</y2>
                  <x3>{x3}</x3>
                  <y3>{y3}</y3>
                  <x4>{x4}</x4>
                  <y4>{y4}</y4>
                </segmentation>
            </object>\n""")
            f.write("</annotation>")

                

    with open(os.path.join(save_dir, "result.txt"), 'w', encoding="utf-8") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    process(src_image_dir, save_dir)