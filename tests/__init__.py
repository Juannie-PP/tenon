# def rotate_identify_and_show_image(
#         small_circle, big_circle, image_type: int = 0,
#         color_type: bool = True, check_pixel: int = 10,
#         similar_precision: int = 2, rotate_type: bool = False,
#         big_circle_empty_radius=None,
#         small_circle_crop_pixel: int = 0, speed_ratio: float = 1,
#         image_show_time: int = 0, proxies=None):
#     try:
#         small_circle_image = image_to_cv2(small_circle, image_type, color_type, proxies)
#         big_circle_image = image_to_cv2(big_circle, image_type, color_type, proxies)
#         if isinstance(small_circle_image, bool) or isinstance(big_circle_image, bool):
#             raise Exception("image_to_cv2 error! 图片解析错误!")
#
#         small_circle_r1 = small_circle_image.shape[:2][1] // 2 - small_circle_crop_pixel
#         small_circle_r2 = small_circle_r1 - check_pixel
#         print(f"small circle -> outer circle: {small_circle_r1}, inner circle: {small_circle_r2}")
#         big_circle_r2 = big_circle_empty_radius if big_circle_empty_radius else small_circle_r1
#         big_circle_r1 = big_circle_r2 + check_pixel
#         print(f"big circle -> outer circle: {big_circle_r1}, inner circle: {big_circle_r2}")
#
#         outer_image_before_resize = mask_image(big_circle_image, big_circle_r1, big_circle_r2)
#         inner_image = mask_image(small_circle_image, small_circle_r1, small_circle_r2)
#         outer_image = cv2.resize(outer_image_before_resize, inner_image.shape[:2])
#
#         similar, total_rotate_angle = rotate_image(inner_image, outer_image,
#         similar_precision, rotate_type)
#
#         cut_small_image = cut_image(small_circle_image, small_circle_r1)
#         height, width = cut_small_image.shape[:2]
#         if rotate_type:
#             rotate_angle = total_rotate_angle
#         else:
#             rotate_angle = -total_rotate_angle
#         mat_rotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), rotate_angle, 1)
#         rotate_small_image = cv2.warpAffine(cut_small_image, mat_rotate, (height, width))
#
#         cv2.imshow('big_circle_image', big_circle_image)
#         cv2.imshow('rotate_small_image', rotate_small_image)
#         inner_rotate_angle = round(total_rotate_angle * speed_ratio / (speed_ratio + 1), 2)
#           if speed_ratio else total_rotate_angle
#         print(f"similar: {similar}, total_angle: {total_rotate_angle},
#         inner_rotate_angle: {inner_rotate_angle}")
#         cv2.waitKey(image_show_time * 1000)
#         return dict(similar=similar, total_angle=total_rotate_angle,
#         inner_angle=inner_rotate_angle)
#     except Exception as e:
#         print('rotate_detect_and_show_image error! :: ' + str(e))
#         return False
