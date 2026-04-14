# CurriculuMoE
### Leveraging Task Compositionality to Effectively Mitigate Catastrophic Forgetting in VLMs
![Incrementally savier magicians](magicians.png) 
---
This repo contains the code developped for the project **CurriculuMoE**.  
Course: [Université de Montréal/Mila IFT-6765](https://admission.umontreal.ca/cours-et-horaires/cours/ift-6765/) - Links between Computer Vision and Language  
Course site: [link](https://sites.google.com/mila.quebec/ift6765-h2026/)  

By:  
[Serge Malo](https://mila.quebec/fr/annuaire/serge-malo)  
[Francis Picard](https://mila.quebec/fr/annuaire/francis-picard)

# Project summary
* VLMs suffer from catastrophic forgetting when continually learning new tasks
* We study how to mitigate this phenomenon in curriculum learning settings for VQA
* We propose CurriculuMoE, a Parameter-Efficient Fine-Tuning approach that incrementally adapts new Experts from existing ones
* We evaluate our approach on Spatial457, a synthetic benchmark for spatial reasoning in VQA
* CurriculuMoE strikes a better balance between stability and plasticity in curriculum learning settings than prior approaches, mitigating catastrophic forgetting while efficiently learning new tasks


# Project results
See our [poster](https://github.com/sergemalo/vlm-cl/blob/main/curriculumoe_poster.pdf) presented on April 16th 2026 at Mila.  
[Examples and references available here.](https://tranquil-metatarsal-615.notion.site/CurriculuMoE-1890c4b9239d80ada024e952b89cdcac?source=copy_link)  


# Model checkpoints
We released our model checkpoints [here](https://drive.google.com/drive/folders/1BEycuOVCK48TX2GT2y1ih_jMu5to99Uz). Each classifier folder contains the corresponding classifier files (the number denotes how many tasks it can detect, starting from task 1). The file moe_adapters.pt contains all Routers and Experts after training for all five tasks.

To validate our results, run the eval_with_moe.py script. For example, to evaluate the performance of our fine-tuned model on level L3_2D_spatial after training on all five tasks, run

```bash
python eval_with_moe.py --level L3_2D_spatial \
  --past_adapters_path {path to moe_adapters.pt} \
  --classifier_path {path to classifier5 folder}
```
---
# Environment setup

## Pre-requisites
1. Python >= 3.11
2. WandB account



## Python Virtual Environment SETUP
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Python Virtual Environment CREATION
Only needed if we need to reset requirements.txt.
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.in
pip freeze > requirements.txt
```

## Python Virtual Environment CHECKS
Try the following commands:
```
python check_pytorch.py
```

And this should work on a 16GB RAM GPU:
```
python eval_llava.py -m 5 -s 5 --use_4bit --save_jsonl  out.txt
```
