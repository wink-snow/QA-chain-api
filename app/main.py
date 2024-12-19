from fastapi import FastAPI
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from interface.app import generate_response_qa_chain
from key import KeyChecker

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

HOST = os.environ["HOST"]
PORT = os.environ["PORT"]

app = FastAPI()

@app.get("/qa_chain/")
async def get_response(user_id: str, question: str):
    """
    This is the API endpoint for the QA chain model.
    """
    checker = KeyChecker(user_id)
    if checker.check() == -1:
        return {"error": "Invalid user id"}
    else:
        response = generate_response_qa_chain(question)
        return {"question": question, "response": response}

if __name__ == "__main__":
    import uvicorn
    print(f"Starting server at {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)