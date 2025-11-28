# MedConnect-AI

## Multi-Agent AI system for Our medical Freelance platform

This agent is built using langraph and uses our finetuned mistral-7B `Agaba-Embedded4/MedConnectAI-FineTunned-4bit` instruct and gemini
---

## Links
* [Continual Pretraining Notebook](https://www.kaggle.com/code/sundayabraham/medconnect-continual-pre-training-mistral)
* [Instruct Tunning Notebook](https://www.kaggle.com/code/burnfroster/instruct-tunning-medconnect)
* [Model Evaluation Notebook](https://www.kaggle.com/code/sundayabraham/evaluating-medconnectai)
---

## Getting Started
1. Clone the repository
```
git clone https://github.com/AgabaEmbedded/MedConnect-AI.git
```
2. create virtual environment
```
virtualenv venv
```
3. activate virtual environment
```
venv\script\activate
```
3. install requirements
```
pip install -r requirement.txt
```
4. set environmental viriables
5. run the script
```
uvicorn main:app --reload
```

---
## System Achitecture
![Multi-Agent AI System Architecture](MedConnect%20Agent%20Architecture.png)

## Evaluation Result
![Evaluation (LLM-AS-A-JUDGE)](llm-as-a-judge.png)

![Evaluation (ROUGE-SCORE)](Rouge-score.png)