from nonebot import logger, require
require("nonebot_plugin_alconna")
from nonebot.adapters import Event, Bot
from nonebot_plugin_alconna import Alconna, on_alconna, UniMessage

import asyncio
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .utils import calculate_average_color, get_avatar
from .typing import Quote
from .protract import draw_quote_segments, append_avatar

FONT_PATH = Path("YaHei Consolas Hybrid 1.12.ttf")
SUB_FONT_PATH = Path("YaHei Consolas Hybrid 1.12.ttf")

quote_cmd = Alconna(".quote")
quote_alc = on_alconna(quote_cmd, aliases={".qt", "qt"})

@quote_alc.handle()
async def handle_command(bot: Bot, event: Event):
    if not event.reply:
        await quote_alc.send(UniMessage.text("请回复一条消息以使用此命令。"))
        return
    user_id, nickname = event.reply.sender.user_id, event.reply.sender.nickname
    if user_id == event.self_id:
        await quote_alc.send(UniMessage.text("Σ( ° △ °|||)︴\n不能回复咱自己的信息"))
        return
    
    avatar_image = await get_avatar(user_id)

    if avatar_image is None:
        await quote_alc.send(UniMessage.text("无法获取用户头像，请检查网络连接。"))
        return
        
    quoteMsg = await Quote.create(event.reply.message)

    image = await process_image(quoteMsg, avatar_image, nickname)
    # image.show()
    if image is not None:
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        await quote_alc.send(UniMessage.image(raw=image_bytes))
    else:
        await quote_alc.send(UniMessage.text("处理图片时发生错误，请检查输入参数。"))
    # await quote_alc.send(UniMessage.text("aa"))


"""async def process_image(quote: Quote, avatar_image: bytes) -> Optional[Image.Image]:
    # loop = asyncio.get_event_loop()
    # async def _process():
        try:
            image = Image.open(BytesIO(avatar_image))
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
            avg_color = calculate_average_color(cv_image)
            print(avg_color)
            
            
            result = await draw_quote_segments(quote, avg_color)

            if result is not None:
                return Image.fromarray(cv2.cvtColor(result, cv2.IMREAD_COLOR))
            else:
                return None
        except Exception as e:
            logger.warning(f"An error occurred during image processing: {e}")
            print(e)
            return None
    # return await loop.run_in_executor(None, _process)"""

async def process_image(quote: Quote, avatar_image: bytes, nickname: str = None) -> Optional[Image.Image]:
    loop = asyncio.get_event_loop()
    
    def _process():
        try:
            image = Image.open(BytesIO(avatar_image))
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
            avg_color = calculate_average_color(cv_image)
            template = cv2.imread("base-2.png",  cv2.IMREAD_UNCHANGED)
            # 里面请使用同步方法
            result = draw_quote_segments(quote, avg_color, (173, 69, 63), "YaHei Consolas Hybrid 1.12.ttf", 42)
            
            result = append_avatar(result, cv_image, (173, 69, 63), "YaHei Consolas Hybrid 1.12.ttf", 28, signature=nickname, template=template)
            
            # result = cv_image

            if result is not None:
                return Image.fromarray(result)
            else:
                return None
        except Exception as e:
            print(e)
            logger.warning(f"An error occurred during image processing: {e}")
            return None
    
    return await loop.run_in_executor(None, _process)



    """ 
    avatar_image = await get_avatar(user_id)

    if avatar_image is None:
        await command.send(MessageSegment.text("无法获取用户头像，请检查网络连接。"))
        return
    
    result_image = await process_image(avatar_image, msg, nickname)

    if result_image:
        image_bytes = BytesIO()
        result_image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        await command.send(MessageSegment.image(image_bytes))
    else:
        await command.send(MessageSegment.text("处理图片时发生错误."))"""


"""
Old Method

def extract_reply_content(reply_msg: Message) -> str:
    reply_content = ""
    for segment in reply_msg:
        if segment.type == "text":
            reply_content += segment.data['text']
        elif segment.type == "image":
            reply_content += segment.data['url']
    return reply_content.strip()


async def process_image(avatar_image: Image.Image, text: str, nickname: str, additional_image: Optional[Image.Image]) -> Optional[Image.Image]:
    loop = asyncio.get_event_loop()

    def _process():
        try:
            png_path = "base0.png"  # base3
            left_margin = 730  #   730  482
            right_margin = 1280  # 1280  960
            bottom_margin = 3
            centre_margin = 0
            extra_margin = 0
            addtion_margin = 0
            png_image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
            if png_image is None:
                logger.warning("PNG Image not found, please check the path.")
                return None

            png_height, png_width, _ = png_image.shape
            target_width, target_height = 720, 720  # 402 402
            start_X, start_Y = 0, 0  # 40 40

            avatar_resized = cv2.resize(cv2.cvtColor(np.array(avatar_image), cv2.COLOR_RGBA2BGRA), (target_width, target_height))
            text_color = calculate_average_color(avatar_resized)
            background = np.zeros((png_height, png_width, 4), dtype=np.uint8)
            background[:, :, :3] = (173, 69, 63)[::-1]
            background[:, :, 3] = 255

            background[start_X:(target_width + start_X), start_Y:(target_height + start_Y), :3] = avatar_resized[:, :, :3]

            alpha_channel = png_image[:, :, 3].astype(float) / 255
            background_float = background.astype(float)

            for channel in range(3):
                mask = alpha_channel > 0
                background_float[mask, channel] = (alpha_channel[mask] * png_image[mask, channel] +
                                               (1 - alpha_channel[mask]) * background_float[mask, channel])

            result = background_float.astype(np.uint8)

        
            if additional_image:
                addtion_margin = -30

                additionalimage = additional_image.convert("RGBA")
                additional_np_array = np.array(additionalimage)
                img_height, img_width, _ = additional_np_array.shape

                max_img_width = (right_margin - left_margin) * 2 // 3
                min_img_width = (right_margin - left_margin) // 3

                max_img_height = png_height * 3 // 4

                max_img_width = (right_margin - left_margin) * 2 // 3
                min_img_width = (right_margin - left_margin) // 3

                scale_factor_height_max = max_img_height / img_height
                scale_factor_width_max = max_img_width / img_width
                scale_factor_width_min = min_img_width / img_width
                if img_width > max_img_width:
                    scale_factor_width = scale_factor_width_max
                elif img_width < min_img_width:
                    scale_factor_width = scale_factor_width_min
                else:
                    scale_factor_width = 1

                scale_factor = min(scale_factor_height_max, scale_factor_width)

                # if img_width < min_img_width:
                #     scale_factor = max(scale_factor, min_img_width / img_width)

                resized_additional_image = cv2.resize(cv2.cvtColor(additional_np_array, cv2.COLOR_RGBA2BGRA), (int(img_width * scale_factor), int(img_height * scale_factor)))

                resized_img_height, resized_img_width, _ = resized_additional_image.shape

                paste_x = left_margin + (right_margin - left_margin - resized_img_width) // 2
                paste_y = png_height // 2 - resized_img_height // 2

                if paste_y + resized_img_height < png_height and paste_x + resized_img_width < png_width:
                    alpha_channel_add = resized_additional_image[:, :, 3].astype(float) / 255
                    for channel in range(3):
                        mask = alpha_channel_add > 0
                        masked_alpha = alpha_channel_add[mask]
                        masked_resized_image = resized_additional_image[mask, channel]
                        masked_result = result[addtion_margin + paste_y:addtion_margin + paste_y + resized_img_height, paste_x:paste_x + resized_img_width, channel][mask]
                        result[addtion_margin + paste_y:addtion_margin + paste_y + resized_img_height, paste_x:paste_x + resized_img_width, channel][mask] = (
                            masked_alpha * masked_resized_image +
                            (1 - masked_alpha) * masked_result
                        )
                extra_margin = 0
                centre_margin = resized_img_height // 2 + addtion_margin

            result = add_main_text(result, text, FONT_PATH, left_margin, right_margin, centre_margin, extra_margin, text_color)

            result = add_nickname(result, nickname, SUB_FONT_PATH, left_margin, right_margin, bottom_margin, 0, 0)


            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))

        except Exception as e:
            logger.warning(e)
            return
    return await loop.run_in_executor(None, _process)


def add_main_text(image: np.ndarray, text: str, font_path: Path, left_margin: int, right_margin: int, centre_margin: int, extra_margin: int = 0, text_color: List = (255, 255, 255)) -> np.ndarray:
    font = ImageFont.truetype(str(font_path), 42)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    draw = ImageDraw.Draw(pil_image)

    text_lines = split_text_to_fit(draw, text, font, right_margin - left_margin)

    total_height = sum([draw.textbbox((0, 0), line, font=font)[3] for line in text_lines])
    total_height += len(text_lines) * 10

    base3_height = pil_image.height
    vertical_center = base3_height // 2
    current_y = vertical_center - total_height // 2 + centre_margin + ((extra_margin + total_height // 2) if centre_margin != 0 else 0)

    for line in text_lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = left_margin + (right_margin - left_margin - text_width) // 2
        draw.text((text_x, current_y), line, fill=text_color, font=font)
        current_y += text_height + 10

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)

def add_nickname(image: np.ndarray, nickname: str, sub_font_path: Path, left_margin: int, right_margin: int, bottom_margin: int, centre_margin: int, extra_margin: int = 0) -> np.ndarray:
    sub_font = ImageFont.truetype(str(sub_font_path), 28)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    draw = ImageDraw.Draw(pil_image)

    sub_text_lines = split_text_to_fit(draw, f"——{nickname}", sub_font, right_margin - left_margin)

    sub_total_height = sum([draw.textbbox((0, 0), line, font=sub_font)[3] for line in sub_text_lines])
    sub_total_height += (len(sub_text_lines) - 1) * 10

    sub_current_y = pil_image.height - bottom_margin - sub_total_height + centre_margin + extra_margin

    for sub_line in sub_text_lines:
        sub_text_bbox = draw.textbbox((0, 0), sub_line, font=sub_font)
        sub_text_width = sub_text_bbox[2] - sub_text_bbox[0]
        sub_text_height = sub_text_bbox[3] - sub_text_bbox[1]
        sub_text_x = left_margin + (right_margin - left_margin - sub_text_width) // 2
        draw.text((sub_text_x, sub_current_y), sub_line, fill=(180, 180, 180), font=sub_font)
        sub_current_y += sub_text_height + 10

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
"""