from pydantic import BaseModel

class Input(BaseModel):
    site_id: str
    user_id: str
    input_text: str
    # topkrelevant: int