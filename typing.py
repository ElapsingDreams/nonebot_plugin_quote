from nonebot.adapters import Message

from typing import List, Dict, Any, Optional
from numpy import ndarray


from .utils import get_image
from .constant import TYPE_TRANSLATION, IGNORE_TYPE, NO_ENTER

class Quote:
    class QuoteSegment:
        # 文本也行
        TEXT = 0
        IMAGE = 1
        MFACE = 2
        AT = 3
        NO_ENTER = 4

        def __init__(self, type_: str, data: str|ndarray, url: Optional[str] = None):
            self.type = type_
            self._data = data
            self.url = url
            self._cached_data = None

        @property
        def get_text(self) -> Optional[str]:
            """方便获取文本类型的数据"""
            if self.type == self.TEXT:  # 确保只有文本类型才尝试获取text
                return self._data if isinstance(self._data, str) else None
            return None
        
        def set_text(self, text: str):
            """
            仅当QuoteSegment为TEXT类型时，设置其数据。
            :param text: 新的文本内容
            """
            if self.type == self.TEXT:
                self._data = text
                # 清除缓存的数据，因为原始数据已经改变
                self._cached_data = None
            else:
                raise ValueError("Attempting to set text for a non-TEXT QuoteSegment")

        @property
        def data(self) -> Dict[str, Any]:
            if self._cached_data is None:
                    self._cached_data = self._load_data() if self.type in (self.IMAGE, self.MFACE) else self._data
            return self._cached_data
        
        def _load_data(self) -> Dict[str, Any]:
            if self.url:
                #print(self.url)
                return get_image(self.url)
            return {}
        
        @classmethod
        async def text(cls, text: str) -> 'Quote.QuoteSegment':
            return cls(cls.TEXT, text)
        
        @classmethod
        async def at(cls, text: str) -> 'Quote.QuoteSegment':
            return cls(cls.AT, text)
        
        @classmethod
        async def no_enter(cls, text: str) -> 'Quote.QuoteSegment':
            return cls(cls.NO_ENTER, text)

        @classmethod
        async def image(cls, url: str) -> 'Quote.QuoteSegment':
            return cls(cls.IMAGE, {}, url)
        
        @classmethod
        async def mface(cls, url: str) -> 'Quote.QuoteSegment':
            return cls(cls.MFACE, {}, url)
        
        """async def __str__(self) -> str:
            data = await self.data
            return f"<Quote type={self.type} data={data}>"

        async def __repr__(self) -> str:
            return await self.__str__()"""

    @classmethod
    async def create(cls, message: Optional[Message]) -> 'Quote':
        instance = cls()
        await instance._parse_message(message)
        return instance

    def __init__(self):
        self.QuoteSegments: List[Quote.QuoteSegment] = []

    async def _parse_message(self, message: Message) -> None:
        no_enter = 0
        for segment in message:
            if segment.type == 'text':
                if no_enter == 1:
                    self.QuoteSegments[-1].set_text(self.QuoteSegments[-1].get_text + segment.data['text'])
                else:
                    self.QuoteSegments.append(await self.QuoteSegment.text(segment.data['text']))
                no_enter = 0
            elif segment.type == 'image':
                # print(segment.data['url'])
                no_enter = 0
                self.QuoteSegments.append(await self.QuoteSegment.image(segment.data['url']))
            elif segment.type == 'mface':
                # print(segment.data['url'])
                no_enter = 0
                self.QuoteSegments.append(await self.QuoteSegment.mface(segment.data['url']))
            elif segment.type == "at":
                self.QuoteSegments.append(await self.QuoteSegment.at("@" + segment.data['name']))
            elif segment.type in NO_ENTER:
                if len(self.QuoteSegments) == 0:
                    self.QuoteSegments.append(await self.QuoteSegment.no_enter("[" + TYPE_TRANSLATION[segment.type] + "]")) 
                elif self.QuoteSegments[-1].type in [self.QuoteSegment.AT, self.QuoteSegment.TEXT, self.QuoteSegment.NO_ENTER]:
                    self.QuoteSegments[-1].set_text(self.QuoteSegments[-1].get_text + " ")
                else:
                    pass
                no_enter = 1
            elif segment.type in IGNORE_TYPE:
                pass
            elif segment.type in TYPE_TRANSLATION.keys():
                self.QuoteSegments.append(await self.QuoteSegment.text("[" + TYPE_TRANSLATION[segment.type] + "]"))
            else:
                self.QuoteSegments.append(await self.QuoteSegment.text("[" + segment.type + "]"))


    """def __str__(self) -> str:
        quotesegments_str = []
        for quote in self.QuoteSegments:
            quotesegments_str.append(quote.__str__())
        return f"<QuoteSegment quotes='{quotesegments_str}'>"

    def __repr__(self) -> str:
        return self.__str__()"""

    def __iter__(self):
        for quote in self.QuoteSegments:
            yield quote
    

    """async def __astr__(self) -> str:
        quotesegments_str = []
        for quote in self.QuoteSegments:
            quotesegments_str.append(quote.__str__())
        return f"<QuoteSegment quotes='{await quotesegments_str}'>"

    async def __arepr__(self) -> str:
        return await self.__astr__()

    async def __aiter__(self):
        for quote in self.QuoteSegments:
            yield quote"""