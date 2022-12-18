import os
path = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnx
import onnxruntime as ort
import argparse


parser = argparse.ArgumentParser('Inference the garbage classification model.')
parser.add_argument('--img_dir', type=str, default='data', help='directory of input images')
args = parser.parse_args()
SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]
FATHER_SRC_PATH = os.path.abspath(os.path.dirname(SRC_PATH) + os.path.sep + ".")
FF_SRC_PATH = os.path.abspath(os.path.join(FATHER_SRC_PATH, ".."))
MODEL_PATH = os.path.join(FF_SRC_PATH, "model"+ os.path.sep + "model_modified.onnx")
MODEL_WIDTH = 224
MODEL_HEIGHT = 224
IMG_EXT = ['.jpg', '.JPG', '.png', '.PNG', '.bmp', '.BMP', '.jpeg', '.JPEG']
image_net_classes = [
   "Seashel", "Lighter","Old Mirror", "Broom","Ceramic Bowl", "Toothbrush","Disposable Chopsticks","Dirty Cloth",
     "Newspaper", "Glassware", "Basketball", "Plastic Bottle", "Cardboard","Glass Bottle", "Metalware", "Hats", "Cans", "Paper",
      "Vegetable Leaf","Orange Peel", "Eggshell","Banana Peel",
    "Battery", "Tablet capsules","Fluorescent lamp", "Paint bucket"]


def get_image_net_class(class_id):
    if class_id >= len(image_net_classes):
        return "unknown"
    else:
        return image_net_classes[class_id]

def pre_process(image):
    """preprocess"""

    resized_image_rgb = image.resize((MODEL_WIDTH, MODEL_HEIGHT))
    resized_image_rgb_np = np.array(resized_image_rgb, dtype=np.float32)
    resized_image_bgr_np = resized_image_rgb_np[:, :, ::-1]
    result = resized_image_bgr_np.transpose((2, 0, 1))
    return result[np.newaxis, :] 


def post_process(infer_output, image_file):
    print("post process")
    data = infer_output[0]
    vals = data.flatten()
    top_k = vals.argsort()[-1:-6:-1]
    object_class = get_image_net_class(top_k[0])
    output_path = os.path.join(os.path.join(SRC_PATH, "../outputs"), os.path.basename(image_file))
    origin_image = Image.open(image_file)
    draw = ImageDraw.Draw(origin_image)
    font = ImageFont.load_default()
    font.size =50
    draw.text((10, 50), object_class, font=font, fill=255)
    origin_image.save(output_path)
    object_class = get_image_net_class(top_k[0])        


def construct_image_info():
    """construct image info"""
    image_info = np.array([MODEL_WIDTH, MODEL_HEIGHT, 
                           MODEL_WIDTH, MODEL_HEIGHT], 
                           dtype = np.float32) 
    return image_info


def modify_graph(onnx_model):
   # modify the opset version
    onnx_model.opset_import[0].version = 10

    # get model graph
    graph = onnx_model.graph

    inode = 0
    # find ReduceMean node
    for idx, node in enumerate(graph.node):
        if "ReduceMean" == node.op_type:
            old_node = node
            inode = idx
            break
    # attributes for new ReduceMean node
    attrs = {
        "axes": node.attribute[0].ints,
        "keepdims": 0,
    }
    # create a new node
    new_node = onnx.helper.make_node(
        op_type="ReduceMean",
        inputs=[node.input[0]],
        outputs=[node.output[0]],
        name=old_node.name,
        **attrs,
    )
    # replace ReduceMean node
    graph.node.remove(old_node)
    graph.node.insert(inode, new_node)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, 'model_modified.onnx')
    return onnx_model

def main():    
    onnx_model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(onnx_model)   
    
    # onnx_model = modify_graph(onnx_model)

    image_dir = args.img_dir
    images_list = [os.path.join(image_dir, img)
                   for img in os.listdir(image_dir)
                   if os.path.splitext(img)[1] in IMG_EXT]

    #Create a directory to store the inference results
    if not os.path.isdir(os.path.join(SRC_PATH, "../outputs")):
        os.mkdir(os.path.join(SRC_PATH, "../outputs"))

    # create onnx inference session for model inference
    ort_sess = ort.InferenceSession(MODEL_PATH)
    for image_file in images_list:        
        # load and resize image
        img = Image.open(image_file)
        img_resized = pre_process(img)
        print("pre process end")
        # ort_sess.get_inputs()[0].name is to get input's name
        result = ort_sess.run([], {ort_sess.get_inputs()[0].name: img_resized})
        post_process(result, image_file)
        print("process "+image_file+" end")
if __name__ == '__main__':
    main()