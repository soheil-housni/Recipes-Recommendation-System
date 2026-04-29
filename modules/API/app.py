import uvicorn
from fastapi import FastAPI
from .recommendation_pipeline import recommendation
from .request_data import RequestData   
from pathlib import Path
import os

app=FastAPI()

@app.get("/")
def welcome():
    return "Welcome to the Recommendation Recipes System"


@app.post("/recommendation")
def recommend_recipes(data:RequestData):
    recommended_recipes=recommendation(user_id=data.user_id,
                                       ratings=data.ratings,
                                       items=data.items,
                                       techniques_users=data.techniques_users,
                                       n_items=data.n_items,
                                       n_ratings=data.n_ratings,
                                       n_recommended_recipes=data.n_recommended_recipes)
    recommended_recipes=recommended_recipes.to_dict(orient="records")
    return recommended_recipes

if __name__=="__main__":
    uvicorn.run(app="modules.API.app:app",host="0.0.0.0",port=8000,reload=False)
    

#uvicorn modules.API.app:app --reload   