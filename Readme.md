# 🚀 NEURO PILOT
AI-powered machine learning workflow automation for everyone

## 📌 Problem Statement
*Problem Statement 11:*  
The AutoML Pipeline Generator is a CLI-based tool designed to automate and optimize machine learning workflows by leveraging AI agents for data preprocessing, feature engineering, model selection, hyperparameter tuning, and performance evaluation. Built on GroqCloud for high-speed processing, it intelligently analyzes datasets, recommends optimal models, and generates deployment-ready pipelines—reducing manual effort while improving accuracy for both beginners and experts in ML.

## 🎯 Objective
NEURO PILOT transforms complex machine learning workflows into intuitive automated processes. By leveraging AI-powered assistance through Groq's high-performance LLMs, our platform democratizes machine learning for data scientists of all skill levels. Users can seamlessly search for datasets, perform intelligent preprocessing, visualize data insights, and automatically train multiple models to find the optimal solution—all through a modern, user-friendly interface rather than traditional CLI commands.

## 🧠 Team & Approach
*Team Name:*  
Bit-Storm Syndicate

*Team Members:*
* Achyut Vyas ([GitHub](https://github.com/snowxx456) | [LinkedIn](https://www.linkedin.com/in/achyut-vyas-874184258/)) – ML Engineer & Backend Developer
* Aryan Khandelwal ([GitHub](https://github.com/flashark271) | [LinkedIn](https://www.linkedin.com/in/aryan10khandelwal/)) – Frontend Developer & UX Designer
* Harsh Jain ([GitHub](https://github.com/Harsh1260) | [LinkedIn](https://www.linkedin.com/in/harsh-jain-b071b424a/)) – Data Scientist & ML Pipeline Architect
* Vaibhav Tayal ([GitHub](https://github.com/vaibhavtayal6) | [LinkedIn](https://www.linkedin.com/in/vaibhavtayal/)) – Full Stack Developer & System Integration


*Our Approach:*
* We identified a significant gap between the powerful capabilities of AutoML and their accessibility. While CLI tools offer flexibility, they often present steep learning curves for newcomers and tedious workflows for experts.
* Our key challenge was transforming a traditionally CLI-based workflow into an intuitive GUI without compromising power or customization. We addressed this by leveraging Groq's LLMs to power intelligent suggestions and automate the entire ML pipeline.
* A major breakthrough occurred when we integrated Groq's API with PandasAI to create a natural language interface for data preprocessing—enabling users to transform their data via conversational prompts rather than complex coding.

## 🛠 Tech Stack
*Core Technologies Used:*
* Frontend: Next.js 14 (TypeScript), React 18, TailwindCSS
* Backend: Python, Django REST Framework
* Database: SQLite
* APIs: Groq, Kaggle

*Sponsor Technologies Used (if any):*
- ✅ **Groq:** _How you used Groq_  
- [ ] **Monad:** _Your blockchain implementation_  
- [ ] **Fluvio:** _Real-time data handling_  
- [ ] **Base:** _AgentKit / OnchainKit / Smart Wallet usage_  
- [ ] **Screenpipe:** _Screen-based analytics or workflows_  
- [ ] **Stellar:** _Payments, identity, or token usage_

## ✨ Key Features
* ✅ *Search Dataset*: Kaggle and Groq integration for direct dataset discovery with download and select options.
* ✅ *Preprocessing*: Fully automated preprocessing through PandasAI and Groq API, covering missing value imputation, encoding, normalization, and more.
* ✅ *Data Visualization and Summary*: Dynamic graphs and charts automatically generated based on the dataset, offering intuitive insights through a user-friendly interface.
* ✅ *Model Training*: Automatic training and evaluation across 15–17 models with metrics like precision, accuracy, and F1-score. Best model selection and downloadable artifacts included.

## 📽 Demo & Deliverables
* *Demo Video Link:* [Demo Video](https://youtu.be/example)
* *PPT Link:* [Presentation](https://docs.google.com/presentation/d/11BqH7XQ8AfJfYi3ePiprhmIqL6-4jijU/edit?usp=sharing&ouid=115184565853677496864&rtpof=true&sd=true)

## ✅ Tasks & Bonus Checklist
* ✅ *All team members completed the mandatory task:* Followed at least two official social channels and submitted the form.
* [ ] *All team members completed Bonus Task 1:* Shared badges and submitted the form (2 points).
* [ ] *All team members completed Bonus Task 2:* Signed up on Sprint.dev and submitted the form (3 points).

## 🧪 How to Run the Project
*Requirements:*
* Python 3.9+
* Node.js 18+
* Groq API key
* Kaggle API key

*Local Setup:*

```bash
# Clone the repository
git clone https://github.com/snowxx456/Neural-Pilot

# Setup backend
cd neuro-pilot/backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Update .env with your API keys

# Run backend server
python manage.py migrate
python manage.py runserver

# Setup frontend (in a new terminal)
cd ../frontend
npm install
npm run dev
