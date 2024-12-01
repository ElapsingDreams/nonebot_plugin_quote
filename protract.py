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
    LINE_SPACING = 10
    # FONT_PATH = "Minecraft像素风格字体(从游戏中提取,支持中英文).ttf"
    total_height = 0 
    segments_images = []

    last_at = None
    for segment in quote:
        if segment.type == Quote.QuoteSegment.TEXT:
            text = ""
            if last_at:
                text += last_at
                last_at = None
            text += segment.data
            font = ImageFont.truetype(font_path, font_size)
            draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))

            text_lines = split_text_to_fit(draw, text, font, MAX_WIDTH)

            text_height = sum([(font.getbbox(line)[3] - font.getbbox(line)[1]) + LINE_SPACING for line in text_lines]) + LINE_SPACING

            text_image = Image.new('RGBA', (MAX_WIDTH, text_height), (173, 69, 63, 0))
            draw = ImageDraw.Draw(text_image)
            
            current_y = 0
            for line in text_lines:
                text_bbox = draw.textbbox((0, 0), line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = (MAX_WIDTH - text_width) // 2  
                draw.text((text_x, current_y), line, fill=textcolor, font=font)
                current_y += (font.getbbox(line)[3] - font.getbbox(line)[1]) + LINE_SPACING
            
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
        elif segment.type == Quote.QuoteSegment.AT:
            last_at = segment.data
    else:
        if last_at:
            text = last_at
            font = ImageFont.truetype(font_path, font_size)
            draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))

            text_lines = split_text_to_fit(draw, text, font, MAX_WIDTH)

            text_height = sum([(font.getbbox(line)[3] - font.getbbox(line)[1]) + LINE_SPACING for line in text_lines]) + LINE_SPACING

            text_image = Image.new('RGBA', (MAX_WIDTH, text_height), (173, 69, 63, 0))
            draw = ImageDraw.Draw(text_image)
            
            current_y = 0
            for line in text_lines:
                text_bbox = draw.textbbox((0, 0), line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = (MAX_WIDTH - text_width) // 2  
                draw.text((text_x, current_y), line, fill=textcolor, font=font)
                current_y += (font.getbbox(line)[3] - font.getbbox(line)[1]) + LINE_SPACING
            
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


def append_avatar(quote_img: np.ndarray, avatar: np.ndarray, bg_color: tuple, font_path: str, font_size: int, avatar_size: tuple = (300, 300), template: np.ndarray = None, margin: int = 50, signature: str = "陌生人"):
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



    # Calculate signature height
    font = ImageFont.truetype(font_path, font_size)
    lines = split_text_to_fit(ImageDraw.Draw(Image.new('RGB', (1, 1))), f'——{signature}', font, image.shape[1])
    signature_height = sum([(font.getbbox(line)[3] - font.getbbox(line)[1]) for line in lines])

    combined_width = w + margin + image.shape[1]
    combined_height = max(h + margin * 2, image.shape[0] + margin //2 + signature_height)
    
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
        y_offset += (font.getbbox(line)[3] - font.getbbox(line)[1])

    signature_cv = np.array(signature_image)
    offset_signature = offset_image + image.shape[0]
    combined_image[offset_signature:offset_signature + signature_height, :image.shape[1]] = signature_cv[:, :, :3]

    return combined_image