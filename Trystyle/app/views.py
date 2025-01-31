from django.shortcuts import render,redirect
from django.views import View
from .models import Customer, Product,Cart, OrderPlaced
from .forms import CustomerRegistrationForm
from django.contrib import messages

#def home(request):
 #return render(request, 'app/home.html')


class ProductView (View):
 def get(self, request):
  topwears = Product.objects.filter(category='TW')
  bottomwears = Product.objects.filter(category='BW')
  return render (request, 'app/home.html', {'topwears':topwears, 'bottomwears':bottomwears})    



#def product_detail(request):
 #return render(request, 'app/productdetail.html')


class ProductDetailView(View):
 def get(self, request, pk):
  product = Product.objects.get(pk=pk)
  return render(request, 'app/productdetail.html', {'product':product})
 
def add_to_cart(request):
 user = request.user 
 product_id = request.GET.get('prod_id')
 product = Product.objects.get(id= product_id)
 Cart(user=user, product = product).save()
 return redirect('/cart')

def show_cart(request):
 if request.user.is_authenticated:
  user = request.user
  cart = Cart.objects.filter(user=user)
  amount = 0.0
  shipping_amount = 70.0
  total_amount = 0.0
  cart_product = [p for p in Cart.objects.all() if p.user == user]
  print (cart_product)

  return render(request, 'app/addtocart.html',{'carts':cart})


def buy_now(request):
 return render(request, 'app/buynow.html')

def profile(request):
 return render(request, 'app/profile.html')

def address(request):
 return render(request, 'app/address.html')

def orders(request):
 return render(request, 'app/orders.html')

def change_password(request):
 return render(request, 'app/changepassword.html')

def mobile(request):
 return render(request, 'app/mobile.html')

class CustomerRegistrationView(View):
 
 def get(self, request):
   form = CustomerRegistrationForm()
   return render(request, 'app/customerregistration.html', {'form':form})
 

 def post(self, request):
    form = CustomerRegistrationForm(request.POST)
    if form.is_valid():
     messages.success(request, 'Congratulations!! Registered Successfully')
     form.save()
    return render(request, 'app/customerregistration.html', {'form':form})


def checkout(request):
 return render(request, 'app/checkout.html')

def tryon(request):
 return render(request, 'app/tryon.html')
