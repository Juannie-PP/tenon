import re
import cv2
import base64
import numpy as np
from typing import Optional


def request_image_content(image_url, proxies: Optional[dict] = None):
    import requests  # type: ignore

    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    }
    response = requests.get(image_url, headers=headers, proxies=proxies)
    return response.content


def set_mask(shape, r1, r2):
    (x, y) = (shape[0] // 2, shape[1] // 2)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask = cv2.circle(mask, (x, y), r1, (255, 255, 255), -1)
    mask = cv2.circle(mask, (x, y), r2, (0, 0, 0), -1)
    return mask


def cut_image(origin_image, crop_pixel):
    height, width = origin_image.shape[:2]
    h_cut_size = (height - crop_pixel * 2) // 2
    w_cut_size = (width - crop_pixel * 2) // 2
    image = origin_image[
        h_cut_size : height - h_cut_size, w_cut_size : width - w_cut_size
    ]
    return image


def mask_image(origin_image, r1, r2):
    deal_image = cut_image(origin_image, r1)
    mask = set_mask(deal_image.shape, r1, r2)
    image = cv2.add(deal_image, np.zeros(deal_image.shape, dtype=np.uint8), mask=mask)
    return image


def rotate_image(inner_image, outer_image, rotate_type):
    angle = 0
    max_val = 0
    start = 0
    end = 360
    h, w = inner_image.shape[:2]
    rotate_type = 1 if rotate_type else -1
    for i in range(0, 2):
        rotate_max_similar = 0
        step = 10 ** (1 - i)
        rotate = start
        while rotate < end:
            rotate += step
            mat_rotate = cv2.getRotationMatrix2D(
                (h * 0.5, w * 0.5), rotate_type * rotate, 1
            )
            dst = cv2.warpAffine(inner_image, mat_rotate, (h, w))
            result = cv2.matchTemplate(outer_image, dst, cv2.TM_CCOEFF_NORMED)
            min_max_loc = cv2.minMaxLoc(result)
            if min_max_loc[1] > max_val:
                max_val = min_max_loc[1]
                if max_val >= rotate_max_similar:  # 获取最高匹配值
                    rotate_max_similar = max_val
                    angle = rotate
        start = angle - step
        end = angle + step
    return max_val, angle


def image_to_cv2(base_image: str, image_type: int, color_type: bool, proxies=None):
    if image_type not in [0, 1, 2]:
        raise Exception("image_type error! 图片类型错误！")
    image_color_type = cv2.COLOR_RGB2GRAY if color_type else cv2.IMREAD_COLOR
    if image_type == 0:
        search_base64 = re.search("base64,(.*?)$", base_image)
        base64_image = search_base64.group(1) if search_base64 else base_image
        image_array = np.asarray(
            bytearray(base64.b64decode(base64_image)), dtype="uint8"
        )
        image = cv2.imdecode(image_array, image_color_type)
    elif image_type == 1:
        image_content = request_image_content(base_image, proxies)
        if not image_content:
            raise Exception("请求图片链接失败！")
        image_array = np.array(bytearray(image_content), dtype=np.uint8)
        image = cv2.imdecode(image_array, image_color_type)
    else:
        image = cv2.cvtColor(cv2.imread(base_image), image_color_type)
    return image


def rotate_identify(
    small_circle: str,
    big_circle: str,
    image_type: int = 0,
    color_type: bool = True,
    check_pixel: int = 10,
    rotate_type: bool = False,
    big_circle_empty_radius: int = 0,
    small_circle_crop_pixel: int = 0,
    speed_ratio: float = 1,
    proxies: Optional[dict] = None,
):
    """
    双图旋转类型滑块验证码识别
    :param small_circle: 小圈图片
    :param big_circle: 大圈图片
    :param image_type: 图片类型: 0: 图片base64; 1: 图片url; 2: 图片文件地址
    :param color_type: 是否需要灰度化处理: True: 是; False: 否
    :param check_pixel: 进行图片验证的像素宽度
    :param rotate_type: 图片旋转的类型: True: 小圈逆时针; False: 小圈顺时针
    :param big_circle_empty_radius: 大圈内部留白部分半径（在留白部分大于内圈实际图片时传值）
    :param small_circle_crop_pixel: 小圈外部留白的像素:（图片宽度 - 有图部分的直径) / 2
    :param speed_ratio: 小圈与大圈的转动速率比: 小圈转动360度时/大圈转动的角度
    :param proxies: 代理
    :return: dict(相似度, 总共旋转的角度, 小圈图片旋转的角度)
    """
    small_circle_image = image_to_cv2(small_circle, image_type, color_type, proxies)
    big_circle_image = image_to_cv2(big_circle, image_type, color_type, proxies)
    if isinstance(small_circle_image, bool) or isinstance(big_circle_image, bool):
        raise Exception("image_to_cv2 error! 图片解析错误!")

    small_circle_r1 = small_circle_image.shape[:2][1] // 2 - small_circle_crop_pixel
    small_circle_r2 = small_circle_r1 - check_pixel

    big_circle_r2 = (
        big_circle_empty_radius if big_circle_empty_radius else small_circle_r1
    )
    big_circle_r1 = big_circle_r2 + check_pixel

    outer_image_before_resize = mask_image(
        big_circle_image, big_circle_r1, big_circle_r2
    )
    inner_image = mask_image(small_circle_image, small_circle_r1, small_circle_r2)
    outer_image = cv2.resize(outer_image_before_resize, inner_image.shape[:2])
    similar, total_angle = rotate_image(inner_image, outer_image, rotate_type)
    inner_angle = round(total_angle * speed_ratio / (speed_ratio + 1), 2)
    return dict(similar=similar, total_angle=total_angle, inner_angle=inner_angle)


def notch_identify(
    slider: str,
    background: str,
    image_type: int = 0,
    color_type: bool = True,
    proxies: Optional[dict] = None,
):
    """
    缺口图片验证码识别
    :param slider: 滑块图片
    :param background: 背景图片
    :param image_type: 图片类型: 0: 图片base64; 1: 图片url; 2: 图片文件地址
    :param color_type: 是否需要灰度化处理: True: 是; False: 否
    :param proxies: 代理, 用于请求图片url链接
    :return: 缺口x轴像素间距
    """
    slider_img = image_to_cv2(slider, image_type, color_type, proxies)
    background_img = image_to_cv2(background, image_type, color_type, proxies)
    if isinstance(slider_img, bool) or isinstance(background_img, bool):
        raise Exception("notch_detect error! 图片解析错误!")
    background_edge = cv2.Canny(background_img, 100, 200)
    slider_edge = cv2.Canny(slider_img, 100, 200)
    background_pic = cv2.cvtColor(background_edge, cv2.COLOR_GRAY2RGB)
    slider_pic = cv2.cvtColor(slider_edge, cv2.COLOR_GRAY2RGB)
    res = cv2.matchTemplate(background_pic, slider_pic, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_loc[0]


def rotate_identify_and_show_image(
    small_circle: str,
    big_circle: str,
    image_type: int = 0,
    color_type: bool = True,
    check_pixel: int = 10,
    rotate_type: bool = False,
    big_circle_empty_radius: int = 0,
    small_circle_crop_pixel: int = 0,
    image_show_time: int = 0,
    proxies: Optional[dict] = None,
):
    small_circle_image = image_to_cv2(small_circle, image_type, color_type, proxies)
    big_circle_image = image_to_cv2(big_circle, image_type, color_type, proxies)
    if isinstance(small_circle_image, bool) or isinstance(big_circle_image, bool):
        raise Exception("image_to_cv2 error! 图片解析错误!")

    small_circle_r1 = small_circle_image.shape[:2][1] // 2 - small_circle_crop_pixel
    small_circle_r2 = small_circle_r1 - check_pixel
    big_circle_r2 = (
        big_circle_empty_radius if big_circle_empty_radius else small_circle_r1
    )
    big_circle_r1 = big_circle_r2 + check_pixel

    outer_image_before_resize = mask_image(
        big_circle_image, big_circle_r1, big_circle_r2
    )
    inner_image = mask_image(small_circle_image, small_circle_r1, small_circle_r2)
    outer_image = cv2.resize(outer_image_before_resize, inner_image.shape[:2])

    similar, total_rotate_angle = rotate_image(inner_image, outer_image, rotate_type)

    cut_small_image = cut_image(small_circle_image, small_circle_r1)
    height, width = cut_small_image.shape[:2]
    if rotate_type:
        rotate_angle = total_rotate_angle
    else:
        rotate_angle = -total_rotate_angle
    mat_rotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), rotate_angle, 1)
    rotate_small_image = cv2.warpAffine(cut_small_image, mat_rotate, (height, width))

    cv2.imshow("big_circle_image", big_circle_image)
    cv2.imshow("rotate_small_image", rotate_small_image)
    cv2.waitKey(image_show_time * 1000)
