import pandas as pd
import uvicorn
import DataParser
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from io import StringIO


app = FastAPI()


@app.post("/file/")
async def load_file(file: UploadFile, list_indexes: str = None):
    # list_indexes = [0, 4]
    list_indexes = list(map(int, list_indexes.split(",")))
    frame = pd.read_csv(file.file)
    dp = DataParser.DataParser()
    frm: pd.DataFrame = dp.sub_parse(frame, list_indexes)
    text_stream = StringIO(frm.to_csv(encoding='utf-8', index=False))
    return StreamingResponse(text_stream, media_type='text/csv')


@app.get("/file/help_list_indexes")
async def file_help_indexes():
    lst = DataParser.DataParser().arrFunctions
    dc = []
    for i, val in enumerate(lst):
        dc.append('index{0} \n'.format(i))
        dc.append('{0} \n'.format(str(val.__doc__)))
    text = ''.join(dc)
    return {"documentation": text}


@app.get("/file/help")
async def file_help():
    return {"documentation": "input file and string of indexes = example[0,1,5] "}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8025)
    # http://127.0.0.1:8025/docs
#if __name__ == '__main__':
    #mlist_indexes = [0, 4]
    #mpath = "course_33c.csv"
    #mframe = pd.read_csv(mpath)
    #mdp = DataParser.DataParser()
    #mfrm: pd.DataFrame = mdp.sub_parse(mframe, mlist_indexes)
    #mfrm.to_csv("file_name.csv", encoding='utf-8', index=False)

