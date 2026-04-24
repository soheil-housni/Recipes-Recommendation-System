from pydantic import BaseModel
from typing import List

class RequestData(BaseModel):
    user_id:int
    ratings:List[int]
    items:List[int]
    techniques_users:List[int]
    n_items:int
    n_ratings:int
    n_recommended_recipes:int=5