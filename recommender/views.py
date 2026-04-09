from django.shortcuts import render
from .ml.predict import generate_wishlist
# Create your views here.
def index(request):
    error = None
    if request.method == "POST":
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        limit = request.POST.get('limit')
        category = request.POST.getlist('category')
        try:
            age_int = int(age)
            limit_int = int(limit)
            if age_int < 1 or age_int > 100:
                error = "Age must be between 1 and 100"
            elif limit_int < 1 or limit_int > 10:
                error = "Limit must be between 1 and 10"
            elif not gender:
                error = "Please select a gender"

            if not error:
                result = generate_wishlist(age_int,gender,category,limit_int)
                return render(request, 'recommender/results.html',{'result':result})
        except (ValueError, TypeError):
            error = "Invalid type, enter a valid number"
           
        
    return render(request, 'recommender/index.html', {'error': error})
    

