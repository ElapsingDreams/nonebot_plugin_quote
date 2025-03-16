from io import BytesIO

import cv2
import imghdr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .typing import Quote
from .utils import split_text_to_fit

def draw_quote_segments(quote: Quote, textcolor: tuple, bg_color: tuple, font_path: str, font_size: int):
    MAX_WIDTH = 600
    MIN_WIDTH = MAX_WIDTH // 3
    LINE_SPACING = 5
    # FONT_PATH = "Minecraft像素风格字体(从游戏中提取,支持中英文).ttf"
    total_height = 0
    segments_images = []
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    last_text = None
    for segment in quote:
        if segment.type == Quote.QuoteSegment.TEXT:
            text = ""
            if last_text:
                text += last_text
                last_text = None
            text += segment.data

            draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))

            text_lines = split_text_to_fit(draw, text, font, MAX_WIDTH)
            # print([(font.getbbox(line)) for line in text_lines])
            text_height = sum([(font.getbbox(line)[3]) + LINE_SPACING for line in text_lines]) # - LINE_SPACING  #  - font.getbbox(line)[1]

            text_image = Image.new('RGBA', (MAX_WIDTH, text_height), bg_color)
            draw = ImageDraw.Draw(text_image)
            
            current_y = 0
            for line in text_lines:
                text_bbox = draw.textbbox((0, 0), line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = (MAX_WIDTH - text_width) // 2  

                '''padding = 0  # 调整边框与文字之间的间距
                bbox = (text_x - padding, current_y - padding, text_x + text_width + padding, current_y + (font.getbbox(line)[3]) + padding)
                draw.rectangle(bbox, outline='yellow', width=1)'''
                draw.text((text_x, current_y), line, fill=textcolor, font=font)
                current_y += (font.getbbox(line)[3]) + LINE_SPACING  # - font.getbbox(line)[1]
            
            segments_images.append(np.array(text_image.convert('RGB')))
            total_height += text_height # + LINE_SPACING
        
        elif segment.type in [Quote.QuoteSegment.IMAGE, Quote.QuoteSegment.MFACE]:
            img_raw = segment.data

            img_type = imghdr.what(None, h=img_raw)
    
            if img_type == 'gif':
                gif_image = Image.open(BytesIO(img_raw))
                gif_image.seek(0)
                png_buffer = BytesIO()
                gif_image.convert('RGBA').save(png_buffer, format='PNG')
                img_raw = png_buffer.getvalue()
                png_buffer.close()

            img_np = cv2.imdecode(np.frombuffer(img_raw, np.uint8), cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_UNCHANGED

            original_height, original_width, axis = img_np.shape

            scale_factor = min(MAX_WIDTH / original_width, 1.0)
            if original_width < MIN_WIDTH:
                scale_factor = max(scale_factor, MIN_WIDTH / original_width)
            
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            img_np = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            img_np[:, :, :3] = img_np[:, :, 2::-1]
            segments_images.append(img_np)
            total_height += new_height
        elif segment.type in [Quote.QuoteSegment.NO_ENTER, Quote.QuoteSegment.AT]:
            last_text = segment.data
    else:
        if last_text:
            text = last_text
            font = ImageFont.truetype(font_path, font_size)
            draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))

            text_lines = split_text_to_fit(draw, text, font, MAX_WIDTH)

            text_height = sum([(font.getbbox(line)[3] ) + LINE_SPACING for line in text_lines]) # - LINE_SPACING  #  - font.getbbox(line)[1]

            text_image = Image.new('RGBA', (MAX_WIDTH, text_height), bg_color)
            draw = ImageDraw.Draw(text_image)
            
            current_y = 0
            for line in text_lines:
                text_bbox = draw.textbbox((0, 0), line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = (MAX_WIDTH - text_width) // 2  
                draw.text((text_x, current_y), line, fill=textcolor, font=font)
                current_y += (font.getbbox(line)[3]) + LINE_SPACING  # - font.getbbox(line)[1]
            
            segments_images.append(np.array(text_image.convert('RGB')))
            total_height += text_height # + LINE_SPACING
    
    result_image = np.zeros((total_height, MAX_WIDTH, 3), dtype=np.uint8)
    result_image[:,:,:3] = bg_color
    current_y = 0
    
    for img_np in segments_images:
        h, w, axis = img_np.shape
        x_offset = (MAX_WIDTH - w) // 2
        if axis == 4:
            alpha_channel = img_np[:, :, 3] / 255.0
            for channel in range(3):
                result_image[current_y:current_y+h, x_offset:x_offset+w, channel] = (
                    alpha_channel * img_np[:, :, channel] +
                    (1 - alpha_channel) * result_image[current_y:current_y+h, x_offset:x_offset+w, channel]
                )
        else: 
            result_image[current_y:current_y+h, x_offset:x_offset+w] = img_np
        current_y += h
    
    return result_image


def append_avatar(quote_img: np.ndarray, avatar: np.ndarray, bg_color: tuple, font_path: str, font_size: int, avatar_size: tuple = (300, 300), 
                template: np.ndarray = None, margin: np.ndarray = [50, 50, 50, 50, 0], signature: str = "陌生人", signature_position: int = 0):
    # margin: np.ndarray = [上, 左, 下, 右, 中, 签]
    margin_up , margin_left, margin_down, margin_right, margin_mid, margin_sign = margin
    h, w, _ = quote_img.shape
    w_a, h_a = avatar_size

    if template is not None:
        h_t, w_t, axis_t = template.shape
        avatar = cv2.resize(avatar, (w_t, h_t), interpolation=cv2.INTER_LANCZOS4)
        image = np.zeros((margin_up + margin_down + h_t, margin_left + w_t, 3), dtype=np.uint8)
        text_width = w_t
    else:
        # h_t, w_t = w_a, h_a
        avatar = cv2.resize(avatar, (w_a, h_a), interpolation=cv2.INTER_LANCZOS4)
        image = np.zeros((margin_up + margin_down + h_a, margin_left + w_a, 3), dtype=np.uint8) 
        text_width = w_a
    image[:,:,:3] = bg_color

    if template is not None:
        centre_y, centre_x = image.shape[0] // 2 - h_t // 2, margin_left

        # image.shape[1] // 2 - w_t // 2  # cancel centre_x
        if axis_t == 4:
            alpha_channel = template[:, :, 3] / 255.0
            for channel in range(3):
                image[centre_y:centre_y + h_t, centre_x:centre_x+w_t, channel] = (
                    alpha_channel * template[:, :, channel] +
                    (1 - alpha_channel) * avatar[:, :, channel]
                )
        else:
            image[centre_y:centre_y + h_t, centre_x:centre_x + w_t] = avatar[:, :, :3]
        image[centre_y:centre_y + h_t, centre_x:centre_x + w_t][:, :, :3] = image[centre_y:centre_y + h_t, centre_x:centre_x + w_t][:, :, 2::-1]
    else:
        centre_y, centre_x = image.shape[0] // 2 - w_a // 2, image.shape[1] // 2 - h_a // 2
        image[centre_y:centre_y + h_a, centre_x:centre_x + w_a] = avatar[:, :, :3]
        image[centre_y:centre_y + h_a, centre_x:centre_x + w_a][:, :, :3] = image[centre_y:centre_y + h_a, centre_x:centre_x + w_a][:, :, 2::-1]

    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    if signature_position == 0:
        lines = split_text_to_fit(ImageDraw.Draw(Image.new('RGB', (1, 1))), f'-- {signature}', font, text_width)
    else:
        lines = split_text_to_fit(ImageDraw.Draw(Image.new('RGB', (1, 1))), f'-- {signature}', font, w)
    signature_line_height = font.getbbox(lines[0])[3]
    signature_height = sum([(font.getbbox(line)[3]) for line in lines])   #  - font.getbbox(line)[1]

    if signature_position == 0:
        combined_height = max(h + margin_up + margin_down + margin_sign, image.shape[0] + signature_height)
        offset_image = 0
        offset_quote = (combined_height - h) // 2
        # offset_signature = image.shape[0] - margin // 2
    else:
        combined_height = max(h + margin_up + margin_down + signature_height + margin_sign, image.shape[0])
        offset_image = 0
        offset_quote = ((combined_height - h - signature_height - signature_line_height) // 2) if h + margin_up + margin_down + signature_height + margin_sign > image.shape[0] + margin_sign else ((combined_height - h - signature_height) // 2)
        # offset_signature = h

    combined_width = w + margin_right + margin_mid + image.shape[1]
    ## combined_height = max(h + margin * 2, image.shape[0] - margin // 4 + signature_height)
    # print(combined_height,h + margin_up + margin_down + signature_height + margin_sign)
    
    ## offset_image = 0  # (combined_height - image.shape[0]) // 2

    combined_image = np.full((combined_height, combined_width, 3), fill_value=bg_color, dtype=np.uint8)

    combined_image[offset_image:offset_image + image.shape[0], :image.shape[1], :] = image
    combined_image[offset_quote:offset_quote + h, image.shape[1] + margin_mid:image.shape[1] + w + margin_mid, :] = quote_img

    if signature_position == 0:
        signature_image = Image.new('RGBA', (text_width, signature_height), bg_color + (0,))
    else:  
        signature_image = Image.new('RGBA', (w, signature_height), bg_color + (0,))
    signature_draw = ImageDraw.Draw(signature_image)

    y_offset = 0
    for line in lines:
        bbox = signature_draw.textbbox((0, 0), line, font=font)
        x_centered = ((image.shape[1] if signature_position == 0 else w) - (bbox[2] - bbox[0])) // 2
        signature_draw.text((x_centered, y_offset), line, font=font, fill=(230, 230, 230))
        y_offset += (font.getbbox(line)[3])  # - font.getbbox(line)[1]

    signature_cv = np.array(signature_image)

    # offset_signature = offset_image + image.shape[0] - margin // 2
    if signature_position == 0:
        # 左
        offset_signature = offset_image + image.shape[0] - margin_down

        combined_image[offset_signature:offset_signature+signature_height, margin_left:margin_left + text_width, :] = signature_cv[:, :, :3]
    else:
        # 右
        offset_signature = offset_quote + h + margin_sign
        combined_image[offset_signature:offset_signature + signature_height, image.shape[1] + margin_mid + 0:image.shape[1] + margin_mid+w] = signature_cv[:, :, :3]

    return combined_image



"""
def append_avatar(quote_img: np.ndarray, avatar: np.ndarray, bg_color: tuple, font_path: str, font_size: int, avatar_size: tuple = (300, 300), template: np.ndarray = None, margin: int = 50, signature: str = "陌生人", signature_position: int = 0):
    h, w, _ = quote_img.shape
    w_a, h_a = avatar_size

    if template is not None:
        h_t, w_t, axis_t = template.shape
        avatar = cv2.resize(avatar, (w_t, h_t), interpolation=cv2.INTER_LANCZOS4)
        image = np.zeros((margin * 2 + h_t, margin * 2 + w_t, 3), dtype=np.uint8)
    else:
       avatar = cv2.resize(avatar, (w_a, h_a), interpolation=cv2.INTER_LANCZOS4)
       image = np.zeros((margin * 2 + h_a, margin * 2 + w_a, 3), dtype=np.uint8) 
    image[:,:,:3] = bg_color

    if template is not None:
        centre_y, centre_x = image.shape[0] // 2 - h_t // 2, image.shape[1] // 2 - w_t // 2
        if axis_t == 4:
            alpha_channel = template[:, :, 3] / 255.0
            for channel in range(3):
                image[centre_y:centre_y + h_t, centre_x:centre_x+w_t, channel] = (
                    alpha_channel * template[:, :, channel] +
                    (1 - alpha_channel) * avatar[:, :, channel]
                )
        else:
            image[centre_y:centre_y + h_t, centre_x:centre_x + w_t] = avatar[:, :, :3]
        image[centre_y:centre_y + h_t, centre_x:centre_x + w_t][:, :, :3] = image[centre_y:centre_y + h_t, centre_x:centre_x + w_t][:, :, 2::-1]
    else:
        centre_y, centre_x = image.shape[0] // 2 - w_a // 2, image.shape[1] // 2 - h_a // 2
        image[centre_y:centre_y + h_a, centre_x:centre_x + w_a] = avatar[:, :, :3]
        image[centre_y:centre_y + h_a, centre_x:centre_x + w_a][:, :, :3] = image[centre_y:centre_y + h_a, centre_x:centre_x + w_a][:, :, 2::-1]



    font = ImageFont.truetype(font_path, font_size)
    lines = split_text_to_fit(ImageDraw.Draw(Image.new('RGB', (1, 1))), f'——{signature}', font, image.shape[1])
    signature_height = sum([(font.getbbox(line)[3]) for line in lines])   #  - font.getbbox(line)[1]

    if signature_position == 0:
        combined_height = max(h + margin * 2, image.shape[0] - margin // 4 + signature_height)
        offset_image = 0
        offset_signature = image.shape[0] - margin // 2
    else:
        combined_height = h + margin * 2 + signature_height
        offset_image = (combined_height - image.shape[0]) // 2
        offset_signature = h + margin


    combined_width = w + margin + image.shape[1]
    combined_height = max(h + margin * 2, image.shape[0] - margin // 4 + signature_height)
    
    offset_quote = (combined_height - h) // 2
    offset_image = 0  # (combined_height - image.shape[0]) // 2

    combined_image = np.full((combined_height, combined_width, 3), fill_value=bg_color, dtype=np.uint8)

    combined_image[offset_image:offset_image + image.shape[0], :image.shape[1], :] = image
    combined_image[offset_quote:offset_quote + h, image.shape[1]:image.shape[1] + w, :] = quote_img


    signature_image = Image.new('RGBA', (image.shape[1], signature_height), bg_color + (0,))
    signature_draw = ImageDraw.Draw(signature_image)

    y_offset = 0
    for line in lines:
        bbox = signature_draw.textbbox((0, 0), line, font=font)
        x_centered = (image.shape[1] - (bbox[2] - bbox[0])) // 2
        signature_draw.text((x_centered, y_offset), line, font=font, fill=(255, 255, 255))
        y_offset += (font.getbbox(line)[3])  # - font.getbbox(line)[1]

    signature_cv = np.array(signature_image)
    offset_signature = offset_image + image.shape[0] - margin // 2
    combined_image[offset_signature:offset_signature + signature_height, :image.shape[1]] = signature_cv[:, :, :3]

    return combined_image

"""