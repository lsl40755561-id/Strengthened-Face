import torch
import numpy as np
import torch.backends.cudnn as cudnn
from PIL import Image

from nets.arcface import Arcface
from ssim_calculate import ssim


if __name__ == "__main__":

    #--------------------------------------#
    cuda            = True
    #--------------------------------------#
    #   mobilefacenet
    #   mobilenetv1
    #   iresnet18
    #   iresnet34
    #   iresnet50
    #   iresnet100
    #   iresnet200
    #--------------------------------------#
    backbone        = "mobilefacenet"

    #--------------------------------------#
    input_shape     = [112, 112, 3]

    #--------------------------------------#
    model_path      = "model_data/arcface_mobilefacenet.pth"


    def resize_image(image, size, letterbox_image):
        iw, ih = image.size
        w, h = size
        if letterbox_image:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
        return new_image


    def get_psnr(x1, x2, range=(-1, 1), reduction=None):
        """ kornia: psnr(x1, x2, float(range[1]-range[0])) """
        mse = torch.sum(torch.square(x2 - x1), dim=(1, 2, 3), keepdim=True) / torch.numel(x1[0])
        psnr = 10 * torch.log10(((range[1] - range[0]) ** 2) / mse)
        if reduction == 'mean':
            psnr = torch.mean(psnr)
        elif reduction == 'sum':
            psnr = torch.sum(psnr)
        return psnr

    def preprocess_input(image):
        image /= 255.0
        return image

    model = Arcface(backbone=backbone, mode="predict")
    th_dict = {'arcface': (0, 0.3409, 0.439)}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model  = model.eval()

    if cuda:
        model = model.to(device)



    # image_original = Image.open('test_face/3/original/0.jpg')
    # image_adv = Image.open('test_face/3/adv/0.jpg')
    # image_strengened = Image.open('test_face/3/strengened/0.jpg')
    # image_strengened_afterattack = Image.open('test_face/3/strengened_afterattack/0.jpg')
    # image_set = Image.open("test_face/3/Aaron_Peirsol_0002.jpg")
    #
    # image_original = Image.open('test_face/5/original/0.jpg')
    # image_adv = Image.open('test_face/5/adv/0.jpg')
    # image_strengened = Image.open('test_face/5/strengened/0.jpg')
    # image_strengened_afterattack = Image.open('test_face/5/strengened_afterattack/0.jpg')
    # image_set = Image.open("test_face/5/Aaron_Peirsol_0002.jpg")

    image_original = Image.open('test_face/8/original/0.jpg')
    image_adv = Image.open('test_face/8/adv/0.jpg')
    image_strengened = Image.open('test_face/8/strengened/0.jpg')
    image_strengened_afterattack = Image.open('test_face/8/strengened_afterattack/0.jpg')
    image_set = Image.open("test_face/8/Aaron_Peirsol_0002.jpg")



    image_original = resize_image(image_original, [input_shape[1], input_shape[0]], letterbox_image=False)
    image_adv = resize_image(image_adv,  [input_shape[1], input_shape[0]], letterbox_image=False)
    image_strengened = resize_image(image_strengened,  [input_shape[1], input_shape[0]], letterbox_image=False)
    image_strengened_afterattack = resize_image(image_strengened_afterattack,  [input_shape[1], input_shape[0]], letterbox_image=False)
    image_set = resize_image(image_set,  [input_shape[1], input_shape[0]], letterbox_image=False)

    image_original, image_adv, image_strengened, image_strengened_afterattack, image_set  = np.transpose(preprocess_input(np.array(image_original, np.float32)), [2, 0, 1]), np.transpose(
        preprocess_input(np.array(image_adv, np.float32)), [2, 0, 1]), np.transpose(preprocess_input(np.array(image_strengened, np.float32)), [2, 0, 1]), np.transpose(
        preprocess_input(np.array(image_strengened_afterattack, np.float32)), [2, 0, 1]), np.transpose(
        preprocess_input(np.array(image_set, np.float32)), [2, 0, 1])

    image_original = torch.from_numpy(image_original).unsqueeze(0).to(device)
    image_adv = torch.from_numpy(image_adv).unsqueeze(0).to(device)
    image_strengened = torch.from_numpy(image_strengened).unsqueeze(0).to(device)
    image_strengened_afterattack = torch.from_numpy(image_strengened_afterattack).unsqueeze(0).to(device)
    image_set = torch.from_numpy(image_set).unsqueeze(0).to(device)

    out_original = model(image_original)
    out_adv = model(image_adv)
    out_strengened = model(image_strengened)
    out_strengened_afterattack = model(image_strengened_afterattack)
    out_set = model(image_set)

    cos_ori = torch.cosine_similarity(out_original, out_set)
    cos_adv = torch.cosine_similarity(out_adv, out_set)
    cos_strengened = torch.cosine_similarity(out_strengened, out_set)
    cos_strengened_afterattack = torch.cosine_similarity(out_strengened_afterattack, out_set)

    ssim_strengened = ssim(image_original, image_strengened)
    ssim_strengened_afterattack = ssim(image_original, image_strengened_afterattack)
    ssim_adv = ssim(image_original, image_adv)


    psnr_strengened = get_psnr(image_original, image_strengened, range=(-1, 1), reduction='sum')
    psnr_strengened_afterattack = get_psnr(image_original, image_strengened_afterattack, range=(-1, 1), reduction='sum')
    psnr_adv = get_psnr(image_original, image_adv, range=(-1, 1), reduction='sum')

    if cos_strengened > th_dict['arcface'][1]:
        recognition_strengend = 1
    else:
        recognition_strengend = 0

    if cos_strengened_afterattack > th_dict['arcface'][1]:
        recognition_strengend_afterattack = 1
    else:
        recognition_strengend_afterattack = 0

    if cos_adv > th_dict['arcface'][1]:
        recognition_adv = 1
    else:
        recognition_adv = 0





    print("Recognition success rate of image_strengened:" + str(recognition_strengend) )
    print("Recognition success rate of image_strengened_afterattack:" + str(recognition_strengend_afterattack))
    print("Recognition success rate of image_adv:" + str(recognition_adv ))
    print("\n")
    print("cosine similarity of image_original: " + f"{cos_ori.item():.4f}")
    print("cosine similarity of image_strengened: " + f"{cos_strengened.item():.4f}")
    print("cosine similarity of image_strengened_afterattack: " + f"{cos_strengened_afterattack.item():.4f}")
    print("cosine similarity of image_adv: " + f"{cos_adv.item():.4f}")







