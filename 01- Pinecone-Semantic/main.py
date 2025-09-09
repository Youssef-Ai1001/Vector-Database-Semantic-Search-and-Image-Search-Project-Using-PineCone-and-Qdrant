# Import Libraries
from fastapi import FastAPI, Form, HTTPException
from utils import delete_vectorDB, insert_vectorDB, search_vectDB

# Initialize an app
app = FastAPI(debug=True)

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI is working 🚀"}

# Endpoint for Searching
@app.post('/semantic_search')
async def semantic_search(search_text:str = Form(...),
                          top_k: int = Form(100),
                          threshold:float = Form(None)):

  # Validation for top_k, and threshold
  if top_k <= 0 or not isinstance(top_k, int) or top_k > 10000 or top_k is None:
    raise HTTPException(status_code=400,
                        detail="Bad Request: 'top_k' must be a positive integer and less than 10000.")

  elif threshold is not None and (threshold <= 0.0 or not isinstance(threshold, float) or threshold > 1.0):
    raise HTTPException(status_code=400,
                        detail="Bad Request: 'threshold' must be a positive float greater than 0.0 and less than 1.0")

  else:
    # Get Similar Records --> Call the (search_vectDB) from utils.py
    similar_records = search_vectDB(query_text=search_text,
                                    top_k=top_k,
                                    threshold=threshold)

  return similar_records


# Endpoint for Updates
@app.post('/update')
async def update(new_text_id: int=Form(...),
                 new_text: str=Form(None),
                 case: str=Form(..., description='case', enum=['upsert', 'delete'])):

    # Validate the new_text is not None if the case=upsert
    if case == 'upsert' and not new_text:
        raise HTTPException(status_code=400,
                            detail='"new_text & class_type" is mandatory for case "upsert".')

    # For Upserting
    if case == 'upsert':
        message = insert_vectorDB(text_id=new_text_id, text=new_text)

    # For Deleting
    elif case == 'delete':
        message = delete_vectorDB(text_id=new_text_id)

    return {'message': message}
