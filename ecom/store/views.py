from django.shortcuts import render

# Create your views here.


def store(request):
    context = {'store':store}
    return render(request,'store/store.html', context)

def cart(request):
    context = {'cart':cart}
    return render(request,'store/cart.html', context)

def checkout(request):
    context = {'checkout':checkout}
    return render(request,'store/checkout.html', context)

def trial(request):
    context= {'trial': trial}
    return render(request,'store/trial.html', context)
    