from fastapi import FastAPI, File, HTTPException, UploadFile
import docx2txt

from fastapi.responses import JSONResponse



from services.findTheme import findKeyWords, transformTopics



app = FastAPI()



@app.get("/")

async def root():

    return {"message": "Hello World"}



@app.post("/upload-doc")

async def upload_doc(file: UploadFile = File(...)):

    if file.filename.lower().endswith('.doc') or file.filename.lower().endswith('.docx'):
        print(file)
        content = docx2txt.process(file.file)

        keywords_metics = findKeyWords(content)
        return {"filename": file.filename, "key_words": transformTopics(keywords_metics)}
    else:
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .doc o .docx")

    



@app.post("/upload-docs")

async def upload_docs(files: list[UploadFile] = File(...)):
    print("files", files)
    arrayTopics = []
    for file in files:

        content_type = file.content_type

        if not file.filename.lower().endswith('.doc') and not file.filename.lower().endswith('.docx'):

            raise HTTPException(status_code=400, detail="Solo se permiten archivos .docx y .doc")
        else:
            content = docx2txt.process(file.file)
            arrayTopics = []
        if(len(arrayTopics) > 0):

            print(arrayTopics)

            #return JSONResponse(content={"message": arrayTopics}, status_code=200)
            return {"key_words": arrayTopics}
        else:
            return {"key_words": []}
    else:
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .doc o .docx")

