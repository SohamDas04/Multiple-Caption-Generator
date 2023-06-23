from summarizer import Summarizer
from PIL import Image
from pytesseract import pytesseract
from flask import Flask,request
from flask import render_template
from flask_cors import CORS
from transformers import logging
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import os
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model_mul = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/result',methods=["POST"])
def result():
    logging.set_verbosity_error()
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image=Image.open(request.files['img']).convert('RGB')
    # conditional image captioning
    text = "A funny"
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    W=processor.decode(out[0], skip_special_tokens=True)

    return W

@app.route('/resultmul',methods=["POST"])
def resultmul():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
 
    raw_image=Image.open(request.files['img']).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    W=processor.decode(out[0], skip_special_tokens=True)

    batch = tokenizer(W,truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
    translated = model_mul.generate(**batch,max_length=60,num_beams=10, num_return_sequences=5, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    Final_String=""
    for cap in tgt_text:
        Final_String='<li class="list-group-item">'+cap+'</li>'+Final_String
    return Final_String

 

 

@app.route("/")
def home():
    return render_template("index.html")

port = int(os.environ.get("PORT", 5000))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
