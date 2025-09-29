import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import torch
from ultralytics import YOLO
import os
import cv2
import numpy as np
from shapely.geometry import Polygon, box
import csv
from datetime import datetime

# --- Global Model Loading ---
cls_model_path = r"C:\Users\anand\Desktop\Crossing\dataset\classification_dataset\Best model - yolo-cls\best (2).pt"
try:
    cls_model = YOLO(cls_model_path)
    # Ensure names attribute exists for classification model
    if not hasattr(cls_model, 'names') or not cls_model.names:
        cls_model.names = {0: "Gates Closed", 1: "Gates Opened"}
    print(f"✅ Classification model loaded from: {cls_model_path}")
except Exception as e:
    messagebox.showerror("Model Load Error", f"Failed to load classification model: {e}")
    exit()

det_model_path = r"C:\Users\anand\Desktop\Crossing\dataset\detection_dataset_split\Output\best_yolov8n_model1\best.pt"
try:
    det_model = YOLO(det_model_path)
    # Ensure names attribute exists for detection model
    if not hasattr(det_model, 'names') or not det_model.names:
        det_model.names = ['bike', 'car', 'cross', 'gate', 'person', 'pole', 'sign_off', 'sign_on', 'train', 'truck', 'wait_plate']
    print(f"✅ Object detection model loaded from: {det_model_path}")
except Exception as e:
    messagebox.showerror("Model Load Error", f"Failed to load object detection model: {e}")
    exit()

# CSV Log File Configuration
CSV_FILE = "railway_monitoring_log.csv"
CSV_HEADERS = [
    "Timestamp", "Gate Location", "Supervisor Name", "Dispatcher Name",
    "Loco Pilot Name", "Train Number", "Direction", "Image Path",
    "Set Gate Status (User)", "Actual Gate Status (Model)",
    "Validation Result", "Obstacles in ROI"
]

# Create CSV file with headers if it doesn't exist, using UTF-8 encoding
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f: # ADDED encoding='utf-8'
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
    print(f"Created new CSV log file: {CSV_FILE}")

class GateStatusApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Task Safety Monitoring System for Railway Crossings Using Computer Vision")
        self.root.geometry("1200x850")

        self.user_selected_status = None
        self.uploaded_image_path = None

        self.original_pil_image = None # Stores the original PIL Image object
        self.processed_pil_image = None # Stores the processed PIL Image object
        self.zoom_factor = 1.0 # Current zoom level
        self.max_img_width = 550
        self.max_img_height = 400

        self.Supervisor_name = "Ramesh S. (ID: GT1021)"
        self.dispatcher_name = "Anil Mehra (ID: DP2034)"
        self.loco_pilot_name = "Shyam Nair (ID: LP5672)"
        self.gate_location = "Gate #12 – Bangalore South"
        self.train_number = "12627 – Karnataka Express"
        self.direction = "NORTH"

        self.entries = {}

        self.create_widgets()

    def create_widgets(self):
        # --- Main Canvas for Scrolling ---
        self.main_canvas = tk.Canvas(self.root)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.main_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        # Bind <Configure> event to update scrollregion when content_frame size changes
        self.main_canvas.bind('<Configure>', lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")))

        # --- Frame to hold all content inside the canvas ---
        self.content_frame = tk.Frame(self.main_canvas)
        # Create a window in the canvas to hold the content frame
        self.main_canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        # Bind mouse wheel for scrolling the canvas itself
        # This lambda ensures the scrollregion is updated when the content frame resizes
        self.content_frame.bind("<Configure>", lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")))
        self.main_canvas.bind_all("<MouseWheel>", self._on_canvas_mousewheel) # Windows/Linux
        self.main_canvas.bind_all("<Button-4>", self._on_canvas_mousewheel) # macOS scroll up
        self.main_canvas.bind_all("<Button-5>", self._on_canvas_mousewheel) # macOS scroll down


        # --- Header Frame (Logo + Title) ---
        header_frame = tk.Frame(self.content_frame)
        header_frame.pack(pady=10, fill=tk.X)

        # Logo at top left
        try:
            logo_img_path = r"C:\Users\anand\Desktop\Crossing\UI\logo3.jpg"
            logo_pil = Image.open(logo_img_path)
            logo_pil.thumbnail((100, 100)) # Resize logo
            self.logo_tk = ImageTk.PhotoImage(logo_pil)
            self.logo_label = tk.Label(header_frame, image=self.logo_tk)
            self.logo_label.pack(side=tk.LEFT, padx=10)
        except FileNotFoundError:
            print(f"❌ Logo file not found at {logo_img_path}. Skipping logo display.")
            self.logo_label = tk.Label(header_frame, text="Logo Missing", font=("Arial", 10))
            self.logo_label.pack(side=tk.LEFT, padx=10)
        except Exception as e:
            print(f"❌ Error loading logo: {e}. Skipping logo display.")
            self.logo_label = tk.Label(header_frame, text="Logo Error", font=("Arial", 10))
            self.logo_label.pack(side=tk.LEFT, padx=10)

        # Application Title
        self.title_label = tk.Label(header_frame, text="Multi-Task Safety Monitoring System for Railway Crossings Using Computer Vision",
                                     font=("Arial", 18, "bold"), fg="#0056b3")
        self.title_label.pack(side=tk.LEFT, padx=20, expand=True, anchor="w")


        # --- Personnel & Location Info Frame ---
        info_frame = tk.Frame(self.content_frame, bd=2, relief="groove")
        info_frame.pack(pady=10, padx=20, fill=tk.X)

        tk.Label(info_frame, text="Supervisor Name:", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=2, sticky="w")
        tk.Label(info_frame, text=self.Supervisor_name, font=("Arial", 10)).grid(row=0, column=1, padx=5, pady=2, sticky="w")

        tk.Label(info_frame, text="Dispatcher Name:", font=("Arial", 10, "bold")).grid(row=1, column=0, padx=5, pady=2, sticky="w")
        tk.Label(info_frame, text=self.dispatcher_name, font=("Arial", 10)).grid(row=1, column=1, padx=5, pady=2, sticky="w")

        tk.Label(info_frame, text="Loco Pilot Name:", font=("Arial", 10, "bold")).grid(row=2, column=0, padx=5, pady=2, sticky="w")
        tk.Label(info_frame, text=self.loco_pilot_name, font=("Arial", 10)).grid(row=2, column=1, padx=5, pady=2, sticky="w")

        tk.Label(info_frame, text="Gate Location:", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=15, pady=2, sticky="w")
        tk.Label(info_frame, text=self.gate_location, font=("Arial", 10)).grid(row=0, column=3, padx=5, pady=2, sticky="w")

        tk.Label(info_frame, text="Train Number:", font=("Arial", 10, "bold")).grid(row=1, column=2, padx=15, pady=2, sticky="w")
        tk.Label(info_frame, text=self.train_number, font=("Arial", 10)).grid(row=1, column=3, padx=5, pady=2, sticky="w")

        tk.Label(info_frame, text="Direction:", font=("Arial", 10, "bold")).grid(row=2, column=2, padx=15, pady=2, sticky="w")
        tk.Label(info_frame, text=self.direction, font=("Arial", 10)).grid(row=2, column=3, padx=5, pady=2, sticky="w")


        # --- Image Upload and Display ---
        self.upload_button = tk.Button(self.content_frame, text="Upload Image", command=self.upload_image, font=("Arial", 12), bg="#e0f2f7")
        self.upload_button.pack(pady=10)

        # Frame for image display (side-by-side)
        self.image_display_frame = tk.Frame(self.content_frame, bd=2, relief="groove")
        self.image_display_frame.pack(pady=10, padx=20, fill=tk.X, expand=True)

        # Original Image Column
        original_img_col = tk.Frame(self.image_display_frame)
        original_img_col.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        tk.Label(original_img_col, text="Original Image", font=("Arial", 10, "bold")).pack()
        self.image_label = tk.Label(original_img_col)
        self.image_label.pack(pady=5, expand=True)
        # Bind mouse scroll events for zoom to the image label
        self.image_label.bind("<MouseWheel>", self._on_mousewheel_zoom) # Windows/Linux
        self.image_label.bind("<Button-4>", self._on_mousewheel_zoom) # macOS scroll up
        self.image_label.bind("<Button-5>", self._on_mousewheel_zoom) # macOS scroll down


        # Processed Image Column
        processed_img_col = tk.Frame(self.image_display_frame)
        processed_img_col.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        tk.Label(processed_img_col, text="Gate Classification and Control Zone Detection", font=("Arial", 10, "bold")).pack()
        self.processed_image_label = tk.Label(processed_img_col)
        self.processed_image_label.pack(pady=5, expand=True)
        # Bind mouse scroll events for zoom to the processed image label
        self.processed_image_label.bind("<MouseWheel>", self._on_mousewheel_zoom) # Windows/Linux
        self.processed_image_label.bind("<Button-4>", self._on_mousewheel_zoom) # macOS scroll up
        self.processed_image_label.bind("<Button-5>", self._on_mousewheel_zoom) # macOS scroll down


        # --- Gate Status Selection ---
        self.gate_status_frame = tk.Frame(self.content_frame)
        self.gate_status_frame.pack(pady=5)

        tk.Label(self.gate_status_frame, text="Set Gate Status:", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=10, pady=5)
        self.gate_open_button = tk.Button(self.gate_status_frame, text="Gate Open", width=15, font=("Arial", 12),
                                          command=lambda: self.set_user_status("Gates Opened"), bg="#d4edda")
        self.gate_open_button.grid(row=0, column=1, padx=5)

        self.gate_closed_button = tk.Button(self.gate_status_frame, text="Gate Closed", width=15, font=("Arial", 12),
                                            command=lambda: self.set_user_status("Gates Closed"), bg="#f8d7da")
        self.gate_closed_button.grid(row=0, column=2, padx=5)

        # --- Validate Button ---
        self.validate_button = tk.Button(self.content_frame, text="Validate", command=self.validate, font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", relief=tk.RAISED)
        self.validate_button.pack(pady=20)

        # --- Result Display ---
        self.result_label = tk.Label(self.content_frame, text="", font=("Arial", 16, "bold"), wraplength=700, justify=tk.CENTER)
        self.result_label.pack(pady=10)

    def _on_canvas_mousewheel(self, event):
        """Handles mouse wheel scrolling for the entire canvas."""
        if event.delta: # Windows/Linux
            self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        elif event.num == 4: # macOS scroll up
            self.main_canvas.yview_scroll(-1, "units")
        elif event.num == 5: # macOS scroll down
            self.main_canvas.yview_scroll(1, "units")

    def _on_mousewheel_zoom(self, event):
        """Handles mouse wheel scrolling specifically for image zoom."""
        # Check if either original or processed image is available for zooming
        if self.original_pil_image is None and self.processed_pil_image is None:
            return

        if event.delta > 0 or event.num == 4: # Scroll up (zoom in)
            self.zoom_factor = min(self.zoom_factor + 0.1, 3.0) # Max zoom factor 3.0
        elif event.delta < 0 or event.num == 5: # Scroll down (zoom out)
            self.zoom_factor = max(0.1, self.zoom_factor - 0.1) # Min zoom factor 0.1
        
        self._update_all_image_displays()

    def _update_image_display(self, pil_image, tk_label):
        """Helper to resize and display a PIL image in a Tkinter Label."""
        if pil_image is None:
            tk_label.config(image='')
            tk_label.image = None # Clear reference
            return

        # Calculate new dimensions based on zoom factor
        original_width, original_height = pil_image.size
        new_width = int(original_width * self.zoom_factor)
        new_height = int(original_height * self.zoom_factor)

        # Ensure minimum size to prevent errors with very small images
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        # Create a temporary image for display, resized according to zoom factor
        display_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Now, thumbnail this potentially larger image to fit within the fixed display area
        display_image.thumbnail((self.max_img_width, self.max_img_height), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(display_image)
        tk_label.config(image=img_tk)
        tk_label.image = img_tk # Keep reference to prevent garbage collection

    def _update_all_image_displays(self):
        """Updates both original and processed image displays with current zoom factor."""
        self._update_image_display(self.original_pil_image, self.image_label)
        self._update_image_display(self.processed_pil_image, self.processed_image_label)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.uploaded_image_path = file_path
            try:
                self.original_pil_image = Image.open(file_path) # Store original PIL image
            except Exception as e:
                messagebox.showerror("Image Load Error", f"Failed to load image: {e}")
                self.original_pil_image = None # Ensure it's None if load fails
                self.uploaded_image_path = None # Clear path as well
                return

            self.zoom_factor = 1.0 # Reset zoom on new image upload
            self._update_image_display(self.original_pil_image, self.image_label) # Display original

            self.result_label.config(text="") # Clear previous results
            # Explicitly clear processed image display and its PIL reference
            self.processed_image_label.config(image='')
            self.processed_image_tk = None
            self.processed_pil_image = None

    def set_user_status(self, status):
        self.user_selected_status = status
        messagebox.showinfo("Gate Status Selected", f"You selected: {status}")

    def validate(self):
        # --- 1. Input Validation ---
        if not self.uploaded_image_path:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        if not self.user_selected_status:
            messagebox.showerror("Error", "Please select the expected gate status (Open/Close).")
            return

        # Gather all data for logging
        log_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Gate Location": self.gate_location,
            "Supervisor Name": self.Supervisor_name,
            "Dispatcher Name": self.dispatcher_name,
            "Loco Pilot Name": self.loco_pilot_name,
            "Train Number": self.train_number,
            "Direction": self.direction,
            "Image Path": os.path.basename(self.uploaded_image_path),
            "Set Gate Status (User)": self.user_selected_status,
            "Actual Gate Status (Model)": "N/A", # Will be updated
            "Validation Result": "N/A", # Will be updated
            "Obstacles in ROI": "N/A" # Will be updated
        }

        # --- 2. Run Classification ---
        try:
            cls_results = cls_model.predict(source=self.uploaded_image_path, imgsz=224, verbose=False)
        except Exception as e:
            messagebox.showerror("Classification Error", f"Error during classification: {e}")
            self.result_label.config(text="Classification Error.", fg="red")
            self.log_to_csv(log_data) # Log the error
            return

        if cls_results and cls_results[0].probs is not None:
            predicted_class_id = int(cls_results[0].probs.top1)
            predicted_class_name = cls_model.names[predicted_class_id]
            log_data["Actual Gate Status (Model)"] = predicted_class_name

            # --- 3. Compare and Determine Validation Result ---
            if predicted_class_name == self.user_selected_status:
                validation_result = "PASS ✅"
                self.result_label.config(fg="green", text=f"PASS \u2705\nPredicted: {predicted_class_name}\nSet: {self.user_selected_status}")
            else:
                validation_result = "FAIL ❌"
                self.result_label.config(fg="red", text=f"FAIL \u274C\nPredicted: {predicted_class_name}\nSet: {self.user_selected_status}")
                messagebox.showwarning("ALERT", "STOP THE TRAIN! Mismatch detected between expected and predicted gate status.")

            log_data["Validation Result"] = validation_result

            # --- 4. Conditional Obstacle Detection (if gates are closed) ---
            if predicted_class_name.lower() == "gates closed":
                self.run_detection_with_dynamic_roi(self.uploaded_image_path, log_data)
            else:
                # If gates are open, no ROI detection needed, just log current state
                self.log_to_csv(log_data)
                # Display the original image in the processed image slot if no detection was performed
                if self.original_pil_image is not None:
                    try:
                        self.processed_pil_image = self.original_pil_image.copy()
                        self._update_image_display(self.processed_pil_image, self.processed_image_label)
                    except Exception as copy_error:
                        # Fallback if copy fails for some reason
                        messagebox.showerror("Image Processing Error", f"Failed to display original image in processed view: {copy_error}")
                        self.processed_pil_image = None
                        self.processed_image_label.config(image='')
                        self.processed_image_tk = None
                else:
                    # If original_pil_image is somehow None, ensure processed display is also cleared
                    self.processed_pil_image = None
                    self.processed_image_label.config(image='')
                    self.processed_image_tk = None
        else:
            self.result_label.config(text="Could not classify the image.", fg="red")
            log_data["Validation Result"] = "ERROR: Classification Failed"
            self.log_to_csv(log_data)

    def run_detection_with_dynamic_roi(self, image_path, log_data):
        try:
            det_results = det_model(image_path)
        except Exception as e:
            messagebox.showerror("Detection Error", f"Error during object detection: {e}")
            log_data["Obstacles in ROI"] = f"ERROR: Detection Failed ({e})"
            self.log_to_csv(log_data)
            return

        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("Image Read Error", "Could not read the image for detection processing.")
            log_data["Obstacles in ROI"] = "ERROR: Image Read Failed"
            self.log_to_csv(log_data)
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB for drawing and display

        gate_boxes = []
        for box_obj in det_results[0].boxes:
            class_id = int(box_obj.cls[0])
            if det_model.names[class_id] == "gate":
                x1, y1, x2, y2 = map(int, box_obj.xyxy[0].tolist())
                gate_boxes.append((x1, y1, x2, y2))

        overlap_found = False
        overlapping_classes = []
        alert_message = ""

        if len(gate_boxes) == 2:
            gate_boxes = sorted(gate_boxes, key=lambda b: b[1])
            g1_x1, g1_y1, g1_x2, g1_y2 = gate_boxes[0]
            g2_x1, g2_y1, g2_x2, g2_y2 = gate_boxes[1] # Corrected this line

            gate1_bl = (g1_x1, g1_y2)
            gate1_br = (g1_x2, g1_y2)
            gate2_tl = (g2_x1, g2_y1)
            gate2_tr = (g2_x2, g2_y1)

            roi_pts = np.array([gate1_bl, gate1_br, gate2_tr, gate2_tl], dtype=np.int32)
            roi_polygon = Polygon(roi_pts)

            for box_obj in det_results[0].boxes:
                class_id = int(box_obj.cls[0])
                class_name = det_model.names[class_id]

                # Skip 'gate' and 'pole' classes for obstacle alerting
                if class_name.lower() == "gate" or class_name.lower() == "pole":
                    continue

                x1, y1, x2, y2 = box_obj.xyxy[0].tolist()
                obj_box = box(x1, y1, x2, y2)

                if roi_polygon.intersects(obj_box):
                    overlap_found = True
                    overlapping_classes.append(class_name)
                    cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3) # Red for alert
                    cv2.putText(img_rgb, f"{class_name}", (int(x1), int(y1) - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Draw detected gate bounding boxes (green)
            for x1, y1, x2, y2 in gate_boxes:
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for gates

            # Draw the dynamic ROI polygon (blue)
            cv2.polylines(img_rgb, [roi_pts], isClosed=True, color=(0, 0, 255), thickness=3) # Blue for ROI
            cv2.putText(img_rgb, "DYNAMIC ROI", (gate1_bl[0], gate1_bl[1] + 20),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if overlap_found:
                alert_message = f"Objects inside ROI: {', '.join(overlapping_classes)}\nSTOP THE TRAIN!"
                messagebox.showwarning("Obstacle Alert", alert_message)
                log_data["Obstacles in ROI"] = ", ".join(overlapping_classes)
            else:
                alert_message = "No objects inside ROI. Safe."
                messagebox.showinfo("Clear", alert_message)
                log_data["Obstacles in ROI"] = "None"

        else:
            alert_message = f"Gate Detection Error: Expected 2 gates to define ROI, but found {len(gate_boxes)}."
            messagebox.showerror("Gate Detection Error", alert_message)
            log_data["Obstacles in ROI"] = f"ERROR: Gate count anomaly ({len(gate_boxes)} found)"

        # Convert processed OpenCV image to Tkinter PhotoImage and display
        self.processed_pil_image = Image.fromarray(img_rgb) # Store processed PIL image
        self._update_image_display(self.processed_pil_image, self.processed_image_label) # Display processed

        # Log all data to CSV
        self.log_to_csv(log_data)

    def log_to_csv(self, data):
        """Appends a row of data to the CSV log file."""
        # Open the CSV file in append mode, specifying UTF-8 encoding
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f: # ADDED encoding='utf-8'
            writer = csv.writer(f)
            # Ensure the row order matches CSV_HEADERS
            row = []
            for header_text in CSV_HEADERS:
                key = header_text.replace(':', '').strip()
                # Special handling for hardcoded values
                if key == "Supervisor Name":
                    row.append(self.Supervisor_name)
                elif key == "Dispatcher Name":
                    row.append(self.dispatcher_name)
                elif key == "Loco Pilot Name":
                    row.append(self.loco_pilot_name)
                elif key == "Gate Location":
                    row.append(self.gate_location)
                elif key == "Train Number":
                    row.append(self.train_number)
                elif key == "Direction":
                    row.append(self.direction)
                else:
                    row.append(data.get(key, 'N/A'))
            writer.writerow(row)
        print(f"Logged data to {CSV_FILE}")

# Main part of the script to run the GUI
if __name__ == '__main__':
    root = tk.Tk()
    app = GateStatusApp(root)
    root.mainloop()
