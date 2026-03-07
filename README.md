# vlm-cl
IFT-6765 Project - Continuous Learning for a VLM  
Serge Malo  
Francis Picard


# Pre-requisites
1. Python >= 3.11



# Python Virtual Environment SETUP
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

# Python Virtual Environment CREATION
Only needed if we break requirements.txt...
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.in
pip freeze > requirements.txt
```

# Python Virtual Environment CHECKS
Try the following commands:
```
python check_pytorch.py
```

And this should work on a 16GB RAM GPU:
```
python eval_llava.py -m 5 -s 5 --use_4bit --save_jsonl  out.txt
```
