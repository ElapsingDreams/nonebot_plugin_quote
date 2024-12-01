from typing import Optional, List

from PIL import Image, ImageDraw, ImageFont
import httpx
import numpy as np


async def async_get_image(url: str) -> Optional[bytes]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        image_data = response.content

        # image = Image.open(BytesIO(image_data))
        # cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)

        return image_data


def get_image(url: str) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    with httpx.Client() as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        image_data = response.content

    return image_data


async def get_avatar(user_id: str) -> Optional[Image.Image]:
    avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
    return await async_get_image(avatar_url)


def split_text_to_fit(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    lines = []
    current_line = ""

    for char in text:
        if char == '\n':
            if current_line:
                lines.append(current_line)
            current_line = ""
        else:
            test_line = f"{current_line}{char}"
            
            test_bbox = draw.textbbox((0, 0), test_line, font=font)
            # print(test_bbox, test_line)
            test_width = test_bbox[2] - test_bbox[0]
            if test_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = char

    if current_line:
        lines.append(current_line)

    return lines


def calculate_average_color(image: np.ndarray) -> tuple:
    average_color_per_row = np.average(image, axis=0)
    average_color = np.average(average_color_per_row, axis=0)
    bgr_mean = average_color[:3]
    return tuple(int(c) for c in bgr_mean)