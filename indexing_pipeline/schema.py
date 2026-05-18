from pydantic import BaseModel, Field

class LectureDocument(BaseModel):
    id:str = Field(alias='objectID')
    title:str
    transcript:str
    url:str
    has_audio:bool = False
    has_video:bool = False
    source:str = "thespiritualscientist"
    video_links: list[str] = []
    audio_links: list[str] = []
    model_config = {
        "populate_by_name": True,
        "str_strip_whitespace": True
    }

class Chunk(BaseModel):
    chunk_id:str
    title:str
    chunk_text:str
    parent_id:str