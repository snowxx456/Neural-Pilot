# 🚀 NEURO PILOT
AI-powered machine learning workflow automation for everyone

## 📌 Problem Statement
*Problem Statement 1:*  
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
* *Demo Video Link:* [Demo Video](https://www.loom.com/share/2899a100803a45e881579c517cc1f2b5?sid=d8dab71b-50bd-4356-9190-9646c79ab574)
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

flowchart TB
    subgraph "User & Presentation Layer"
        Browser["User Browser"]:::ui
        Frontend["Next.js Frontend"]:::ui
    end

    subgraph "API Layer"
        API["Django REST API"]:::backend
        Settings["Django Config"]:::backend
        Urls["URL Routing"]:::backend
        WSGI["WSGI Entry"]:::backend
        Manage["manage.py"]:::backend
    end

    subgraph "ML Pipeline Modules"
        subgraph "Data Cleaning"
            DataCleaning["Data Cleaning Module"]:::ml
            LLMBase["LLM Agent Base"]:::ml
            LLMConfig["LLM Config"]:::ml
            LLMScript["LLM Cleaning Script"]:::ml
        end
        subgraph "Data Visualization"
            DataViz["Data Visualization Module"]:::ml
        end
        subgraph "Model Training"
            ModelTrain["Model Training Module"]:::ml
        end
    end

    subgraph "Storage & Database"
        Media["File Storage (media/)"]:::storage
        Uploads["Uploads (datasets, cleaned_datasets)"]:::storage
        DB["SQLite Database"]:::storage
    end

    subgraph "External Services"
        GroqClient["Groq LLM API"]:::external
        KaggleConfigNode["Kaggle Config"]:::external
        KaggleJSON["Kaggle Credentials"]:::external
        Downloads["Kaggle Downloads"]:::external
    end

    Browser -->|"HTTP/HTTPS"| Frontend
    Frontend -->|"REST API (JSON)"| API
    API -->|"Internal Python Calls"| DataCleaning
    API -->|"Internal Python Calls"| DataViz
    API -->|"Internal Python Calls"| ModelTrain
    DataCleaning -->|"pickle read/write"| Media
    DataCleaning -->|"pickle read/write"| Uploads
    DataViz -->|"load charts data"| Media
    ModelTrain -->|"store models"| Media
    API -->|"SQL Queries"| DB
    DataCleaning -->|"PandasAI API Call"| GroqClient
    API -->|"Kaggle API"| Downloads

    click Frontend "https://github.com/snowxx456/neural-pilot/tree/main/Frontend/"
    click API "https://github.com/snowxx456/neural-pilot/tree/main/Backend/api/"
    click Settings "https://github.com/snowxx456/neural-pilot/blob/main/Backend/config/settings.py"
    click Urls "https://github.com/snowxx456/neural-pilot/blob/main/Backend/config/urls.py"
    click WSGI "https://github.com/snowxx456/neural-pilot/blob/main/Backend/config/wsgi.py"
    click Manage "https://github.com/snowxx456/neural-pilot/blob/main/Backend/manage.py"
    click DataCleaning "https://github.com/snowxx456/neural-pilot/blob/main/Backend/model/data_cleaning/data_preprocessing.py"
    click LLMBase "https://github.com/snowxx456/neural-pilot/blob/main/Backend/model/data_cleaning/llm/agent/base.py"
    click LLMConfig "https://github.com/snowxx456/neural-pilot/blob/main/Backend/model/data_cleaning/llm/config.py"
    click LLMScript "https://github.com/snowxx456/neural-pilot/blob/main/Backend/llm_clean_data_script.py"
    click DataViz "https://github.com/snowxx456/neural-pilot/blob/main/Backend/model/data_visualization/data_visualization.py"
    click ModelTrain "https://github.com/snowxx456/neural-pilot/tree/main/Backend/model/modeltraining/"
    click Media "https://github.com/snowxx456/neural-pilot/tree/main/Backend/media/"
    click Uploads "https://github.com/snowxx456/neural-pilot/tree/main/Backend/uploads/"
    click GroqClient "https://github.com/snowxx456/neural-pilot/blob/main/Backend/model/search/groq_client.py"
    click KaggleConfigNode "https://github.com/snowxx456/neural-pilot/blob/main/Backend/model/search/config_secrets.py"
    click KaggleJSON "https://github.com/snowxx456/neural-pilot/blob/main/Backend/model/search/kaggle.json"
    click Downloads "https://github.com/snowxx456/neural-pilot/tree/main/Backend/model/search/downloads/"

    classDef ui fill:#D0E8FF,stroke:#3399FF,color:#000000;
    classDef backend fill:#DFF0D8,stroke:#5CB85C,color:#000000;
    classDef ml fill:#EAE0F8,stroke:#8A2BE2,color:#000000;
    classDef storage fill:#EFEFEF,stroke:#AAAAAA,color:#000000;
    classDef external fill:#FFE5CC,stroke:#FF9933,color:#000000;

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
