# 🎁 AI Wishlist Generator
 
A machine learning-powered birthday wishlist generator that recommends personalised gifts based on age, gender, and category preferences.
 
## Live Demo
[ai-wishlist-generator.vercel.app](https://ai-wishlist-generator.vercel.app)
 
---
 
## About
AI Wishlist Generator takes a few simple inputs from the user and uses a trained Support Vector Machine (SVM) model to predict and return a personalised list of gift recommendations. The dataset was custom-built and covers gift items across 6 categories suitable for ages 1 to 100.
 
---
 
## How It Works
1. User fills in their age, gender, preferred categories and how many gifts they want
2. The inputs are passed to a trained SVM model
3. The model predicts which gifts from the dataset are relevant for that profile
4. A shuffled, limited wishlist is returned and displayed
5. If the model returns no results, a fallback filter ensures the user always gets recommendations
 
---
 
## Features
- SVM-based gift recommendation engine
- Custom gift dataset with 296 items across 6 categories
- Age coverage from 1 to 100
- Fallback filtering when model confidence is low
- Clean, mobile-friendly UI built with Tailwind CSS
- Input validation on both frontend and backend
 
---
 
## Tech Stack
- **Backend** — Python, Django
- **Machine Learning** — scikit-learn (SVM), pandas, numpy, joblib
- **Frontend** — Django Templates, Tailwind CSS
- **Deployment** — Vercel
 
---
 
## Dataset
The gift dataset was custom-built and contains 296 gift items across 6 categories:

| Category | Description |
|---|---|
| Tech | Gadgets and electronic devices |
| Lifestyle & Assets | Investments, household items, life essentials |
| Beauty & Fashion | Clothing, accessories, skincare, body care |
| Leisure & Sentiment | Books, personalised gifts, experiences |
| Entertainment | Games, toys, sports, creative kits |
| Consumables | Food, drinks, snacks, hampers |

Each item has a gender suitability (male, female, unisex) and an age range (min_age, max_age).

 
---
 
## ML Approach
- **Model**: Support Vector Machine (SVM) with linear kernel
- **Future plan**: Collect real user feedback via Google Form → expand dataset → retrain model → layer in K-Means clustering for user profiling
 
---
 
## Installation
 
```bash
# Clone the repo
git clone https://github.com/Bernadettechukwuedo/AI-WISHLIST-GENERATOR.git
cd AI-WISHLIST-GENERATOR
 
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
 
# Install dependencies
pip install -r requirements.txt
 
# Run the development server
python manage.py runserver
```
 
---
 
## Project Structure
```
AI-WISHLIST-GENERATOR/
├── core/                   # Django project settings
├── recommender/            # Main app
│   ├── ml/
│   │   ├── gift_dataset.csv
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── model.pkl
│   │   └── le_gender.pkl
│   ├── templates/
│   │   └── recommender/
│   ├── views.py
│   └── urls.py
├── manage.py
├── requirements.txt
└── vercel.json
```
 
---
 
## Contributing
> For the model to learn better, a larger dataset is needed. Got a gift idea? Please fill in this form — it takes less than a minute!
> 👉 [Contribute to the dataset](https://forms.gle/XFCWEsJxWikHvEAg6)
 
---
 
## Author
**Bernadette Chukwuedo**
Built as a birthday project — April 2026