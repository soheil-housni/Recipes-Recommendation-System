import uvicorn
from fastapi import FastAPI
from .recommendation_pipeline import recommendation
from .request_data import RequestData   

app=FastAPI()

@app.get("/")
def welcome():
    return "Welcome to the Recommendation Recipes System"


@app.get("/recommendation")
def prediction(data:RequestData):
    recommended_recipes=recommendation(user_id=data.user_id,
                                       ratings=data.ratings,
                                       items=data.items,
                                       techniques_users=data.techniques_users,
                                       n_items=data.n_items,
                                       n_ratings=data.n_ratings,
                                       n_recommended_recipes=data.n_recommended_recipes)
    print(recommended_recipes)
    return recommended_recipes
    

if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

#uvicorn main:app --reload   