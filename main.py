import torch
from dataset import *
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from pathlib import Path
import io

SUPPORTED_EXTENSIONS = {"jpeg", "jpg", "png"}
SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("saved_weights/unet_GAN_25k", map_location=device)

def denormalize(img_norm):
    img_norm[:, :, 0] = (img_norm[:, :, 0] + 1.) * 50.
    img_norm[:, :, 1] = ifunc(img_norm[:, :, 1])
    img_norm[:, :, 2] = ifunc(img_norm[:, :, 2])
    return img_norm

# Start the app
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=Path(__file__).parent.absolute() / "static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Inference endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    
    # Check if the image file is valid
    filename = file.filename
    if '.' not in filename:
        raise HTTPException(status_code=300, detail="Image file must have an extension")
    
    file_extension = filename.split('.')[-1]
    if file_extension not in SUPPORTED_EXTENSIONS:
        error_msg = "Image file extension '" + file_extension + "' is not supported. Please use one of the following: " + str(SUPPORTED_EXTENSIONS)
        raise HTTPException(status_code=301, detail=error_msg)
        
    # Read the image file and predict the color
    content = await file.read()
    img = Image.open(io.BytesIO(content))
    img = img.resize((SIZE, SIZE))
    img = np.array(img)
    try:
        img_lab = rgb2lab(img).astype("float32")

    except ValueError:
        img = gray2rgb(img)
        img_lab = rgb2lab(img).astype("float32")

    img_lab = transforms.ToTensor()(img_lab)
    
    L = img_lab[[0], ...] / 50. - 1.
    ab = func(img_lab[[1, 2], ...])
    
    L = L.unsqueeze(0).to(device)
    ab = ab.unsqueeze(0).to(device)
    
    colorized = model(L)
    result = torch.cat([L, colorized], dim=1)
    result = result.squeeze(0).permute(1, 2, 0)
    result = result.detach().cpu().numpy()
    result = lab2rgb(denormalize(result))
    result = Image.fromarray((result * 255).astype(np.uint8))
    
    # Save image to an in-memory bytes buffer
    with io.BytesIO() as buffer:
        result.save(buffer, format='JPEG')
        result_bytes = buffer.getvalue()
    
    headers = {'Content-Disposition': 'inline; filename="result.jpg"'}
    return Response(result_bytes, headers=headers, media_type='image/jpg')
    
    
    
