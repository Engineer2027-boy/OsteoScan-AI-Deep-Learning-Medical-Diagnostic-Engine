Title : OsteoScan AI:Deep Learning Medical Diagnostic Engine    
____________________________________________________________________________________________________________________________________________________________________________________________________________________
OsteoScan AI is a full-stack medical diagnostic platform built with PyTorch, FastAPI, and React.js, powered by a fine-tuned 121-layer DenseNet architecture. Designed as a real-time clinical triage assistant,    
the system evaluates chest X-rays to accelerate emergency room workflows. It achieves a 6.8% diagnostic error rate and 92.4% sensitivity on unseen test benchmarks, significantly outperforming average clinical    
radiologist error rates of ~17.5%. The core novelty of the project lies in its integration of Explainable AI (XAI) using Class Activation Maps (CAM), which project spatial heatmaps over high-risk pathology    
regions directly within the UI for transparent decision support.    
____________________________________________________________________________________________________________________________________________________________________________________________________________________
Features    
•Engineered a full-stack diagnostic platform using PyTorch, FastAPI, and React.js, leveraging a fine-tuned 121-layer DenseNet backbone.    

•Achieved a 6.8% diagnostic error rate and 92.4% sensitivity on test benchmarks (outperforming the ~17.5% average clinical radiologist error rate) to accelerate emergency triage.    

•Novelty: Integrated Explainable AI (XAI) via real-time Class Activation Maps (CAM) directly in the UI, rendering spatial heatmaps over high-risk pathology regions for transparent diagnostic decision support.    
____________________________________________________________________________________________________________________________________________________________________________________________________________________
Project Structure
_________________
├── backend/    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── comparision.py    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── main.py         
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── model.py    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── eval_results.json     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── train.py    
├── frontend/        
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── package.json    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── package-lock.json    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── postcss.config.js    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── tailwing.config.js    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── vite.config.js    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── src/    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── App.jsx    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── index.css    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── main.jsx    
├── requirements.txt # Python dependencies  
├── README.md # (this file) 
____________________________________________________________________________________________________________________________________________________________________________________________________________________
License  
[ISC](https://github.com/Engineer2027-boy/OsteoScan-AI-Deep-Learning-Medical-Diagnostic-Engine/blob/main/LICENSE)
____________________________________________________________________________________________________________________________________________________________________________________________________________________
Author  
Ayush Aditya
