from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.contrib.auth.views import LoginView, PasswordResetView, PasswordChangeView
from django.contrib import messages
from django.contrib.messages.views import SuccessMessageMixin
from django.views import View
from django.contrib.auth.decorators import login_required

from .forms import RegisterForm, LoginForm, UpdateUserForm, UpdateProfileForm
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
from PIL import Image
import json
from datetime import datetime

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

def home(request):
    return render(request, 'users/home.html')


class RegisterView(View):
    form_class = RegisterForm
    initial = {'key': 'value'}
    template_name = 'users/register.html'

    def dispatch(self, request, *args, **kwargs):
        # will redirect to the home page if a user tries to access the register page while logged in
        if request.user.is_authenticated:
            return redirect(to='/')

        # else process dispatch as it otherwise normally would
        return super(RegisterView, self).dispatch(request, *args, **kwargs)

    def get(self, request, *args, **kwargs):
        form = self.form_class(initial=self.initial)
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)

        if form.is_valid():
            form.save()

            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}')

            return redirect(to='login')

        return render(request, self.template_name, {'form': form})


# Class based view that extends from the built in login view to add a remember me functionality
class CustomLoginView(LoginView):
    form_class = LoginForm

    def form_valid(self, form):
        remember_me = form.cleaned_data.get('remember_me')

        if not remember_me:
            # set session expiry to 0 seconds. So it will automatically close the session after the browser is closed.
            self.request.session.set_expiry(0)

            # Set session as modified to force data updates/cookie to be saved.
            self.request.session.modified = True

        # else browser session will be as long as the session cookie time "SESSION_COOKIE_AGE" defined in settings.py
        return super(CustomLoginView, self).form_valid(form)


class ResetPasswordView(SuccessMessageMixin, PasswordResetView):
    template_name = 'users/password_reset.html'
    email_template_name = 'users/password_reset_email.html'
    subject_template_name = 'users/password_reset_subject'
    success_message = "We've emailed you instructions for setting your password, " \
                      "if an account exists with the email you entered. You should receive them shortly." \
                      " If you don't receive an email, " \
                      "please make sure you've entered the address you registered with, and check your spam folder."
    success_url = reverse_lazy('users-home')


class ChangePasswordView(SuccessMessageMixin, PasswordChangeView):
    template_name = 'users/change_password.html'
    success_message = "Successfully Changed Your Password"
    success_url = reverse_lazy('users-home')


@login_required
def profile(request):
    if request.method == 'POST':
        user_form = UpdateUserForm(request.POST, instance=request.user)
        profile_form = UpdateProfileForm(request.POST, request.FILES, instance=request.user.profile)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Your profile is updated successfully')
            return redirect(to='users-profile')
    else:
        user_form = UpdateUserForm(instance=request.user)
        profile_form = UpdateProfileForm(instance=request.user.profile)

    return render(request, 'users/profile.html', {'user_form': user_form, 'profile_form': profile_form})

@login_required

@csrf_exempt
def livestream(request):
    if request.method == 'POST':
        # Get user-drawn bounding box coordinates, IP address, and screen name from the request
        bounding_boxes_str = request.POST.get('boxes')
        ip = request.POST['ip']
        screen = request.POST['screen']
        print("Screen: ", screen)
        print(bounding_boxes_str)
        
        # Convert the bounding_boxes string to a dictionary
        bounding_boxes = json.loads(bounding_boxes_str)

        # Initialize a list to store the detection results
        detection_results = []

        # Iterate through bounding box coordinates
        for box in bounding_boxes:
            # Assign x1, y1, x2, y2 from the current bounding box coordinate
            x1, y1, x2, y2 = box

            # Open the video stream
            video_path = f"http://{ip}"
            detected_list = []
            box_list = []
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return render(request, 'livestream.html', {'error': 'Error opening video stream!'})

            # Initialize detection flag
            detected = False

            # Read the frame from the video stream
            ret, frame = cap.read()

            # Check if a frame is read successfully
            if not ret:
                break

            # Crop the frame to the selected bounding box
            danger_zone = frame[y1:y2, x1:x2]

            # Convert the cropped frame to PIL Image
            pil_image = Image.fromarray(danger_zone)

            # Perform object detection using YOLOv8
            results = model(pil_image)
            # print(results)
            # labels = results.xyxyn[0].cpu().numpy()[:, -1]
            result = next(iter(results))
            labels = result.boxes.cls.cpu().numpy()
            classes = model.names[int(labels[0])] if len(labels) > 0 else None

            # Check if a person is detected in the danger zone
            if classes == 'person':
                detected = True
                detected_list.append(detected)
                box_list.append(box)
                print("TRUE ===", box, "IP ===", ip)
                log_detection(ip, screen)  # Call the log_detection() function to store the detection information
            else:
                detected_list.append(detected)
                box_list.append(box)
                print("FALSE ===", box, "IP ===", ip)

            # Add the detection result to the list
            detection_results.append({'detected': detected, 'box': box})
            # print(detection_results)
            # print(type(detection_results))

            cap.release()

        results = [{'detected': detected, 'box': box, 'ip': ip} for detected, box in zip(detected_list, box_list)]
        # Return the detection results as JSON response
        return JsonResponse({'results': detection_results})

    return render(request, 'users/livestream.html')


def log_detection(ip, screen):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": current_datetime,
        "Person Detected":True,
        "ip_address": ip,
        "screen": screen
    }
 
    log_file_path = "log_file.json"  

    # Load existing log entries from the file
    existing_logs = []
    try:
        with open(log_file_path, "r") as log_file:
            existing_logs = json.load(log_file)
    except FileNotFoundError:
        pass

    # Append the new log entry to the existing logs
    existing_logs.append(log_entry)

    # Write the updated log entries to the file
    with open(log_file_path, "w") as log_file:
        json.dump(existing_logs, log_file)
